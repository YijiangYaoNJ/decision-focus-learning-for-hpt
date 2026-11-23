"""Lock April-only analogues, a historical actuator vector and May 600-second episodes."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ieee123_ramp_windows import ARCHIVE, ROOT, SIX, day_data

OUT = ROOT/"runs/ieee123_interval_600_v1"
MAY = ARCHIVE.with_name("oahu_201005.zip")
OLD = ROOT/"runs/ieee123_fast_ramps_v1"
CAP = np.array([300,300,500,500,500,500])
PRIOR = 10


def phase_power(p):
    return np.repeat(p*CAP/3,3,axis=-1)


def spatial(p):
    d=p[5:]-p[:-5]
    energy=np.sqrt(np.mean(d*d))
    direction=d.mean(1)
    coherence=np.mean(direction**2)/max(np.mean(d*d),1e-12)
    pos=np.maximum(0,d).mean(1)
    neg=np.maximum(0,-d).mean(1)
    opposition=2*np.minimum(pos,neg).mean()/max((pos+neg).mean(),1e-9)
    norm=(d-d.mean(0))/np.maximum(d.std(0),1e-6)
    tri=np.triu_indices(6,1)
    same=(norm.T@norm/len(d))[tri]
    best=same.copy()
    for k in (2,4,6,8):
        cross=norm[k:].T@norm[:-k]/(len(d)-k)
        best=np.maximum(best,np.maximum(cross[tri],cross.T[tri]))
    return dict(ramp=energy,synchronous=coherence,opposing=opposition,lagged=(best-same).mean())


def feature(p):
    return np.r_[p[-1],3*(p[-1]-p[-6]),1.5*(p[-6]-p[-11])]


def forecasts(tree,lib,history):
    # Two measured joint continuations and persistence. Choose central and
    # dispersed paths among the 32 closest April histories, using no May future.
    _,idx=tree.query(feature(history),k=32)
    a=lib[idx]
    mid=np.argmin(np.mean((a-np.median(a,axis=0))**2,axis=(1,2)))
    far=np.argmax(np.mean((a-a[mid])**2,axis=(1,2)))
    increments=np.stack([np.zeros_like(a[0]),a[mid],a[far]])
    return np.clip(history[-1]+increments,0,1)


def select():
    from ieee123_fast_ramps import predictor
    from ieee123_service_areas import build,metrics,local,ZERO
    OUT.mkdir(parents=True,exist_ok=True)
    features=[]; futures=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in sorted(z.namelist()):
            p=day_data(z,name)[:,SIX]
            for t in range(36000,50400,30):
                block=p[t-10:t+6]
                if np.isfinite(block).all():
                    features.append(feature(block[:11]))
                    futures.append(block[10:16]-block[10])
    np.savez_compressed(OUT/"development.npz",features=features,futures=futures)
    with np.load(OLD/"commands.npz") as c:
        fixed=c["20100407_0_AC-OPF_0"]
    assert fixed.shape==(54,)
    records=[]; data={}
    with zipfile.ZipFile(MAY) as z:
        for name in sorted(z.namelist()):
            p=day_data(z,name)
            day=name[:8]
            for boundary in range(36000,50400,900):
                t=boundary+120
                h=p[boundary-3600:boundary].reshape(4,900,9)
                block=p[t-PRIOR:t+600,SIX]
                if not (np.isfinite(h).all() and np.isfinite(block).all()): continue
                slot=(datetime.strptime(day,"%Y%m%d").timetuple().tm_yday-1)*96+t//900
                hist=np.r_[h.mean(1)[::-1].ravel(),np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96)]
                records.append(dict(date=day,second=t,slot=slot,**spatial(block[PRIOR:])))
                data[slot]=(hist,block)
    slow,hashes=predictor(np.stack([data[r["slot"]][0] for r in records]))
    _,model,*_=build()
    for row,s in zip(records,slow):
        slot=row["slot"]
        st=model.step(slot,ZERO,"high",generation_kw=phase_power(s))
        initial=local(st,model,ZERO)
        prior=model.solve(slot,initial,"high",st["taps"],generation_kw=phase_power(data[slot][1][PRIOR-1]),
                          tolerance=1e-8,max_iterations=250)
        m=metrics(model,prior,slot,"high","selection",model.ratings.ravel())
        row.update(device_headroom=1-max(m["main_util"],m["port_util"]),
                   voltage_headroom=min(m["vmin"]-.95,1.05-m["vmax"]))
        data[slot]+= (initial,np.asarray(st["taps"]),s)
    df=pd.DataFrame(records)
    df.to_csv(OUT/"eligible.csv",index=False)
    # One uniform natural window, then distinct dates in the upper quartile
    # of each spatial score, targeting low/high observed initial headroom.
    natural=df.sample(n=1,random_state=123).iloc[0]
    selected=[dict(**natural.to_dict(),pattern="natural",reserve="random")]
    used={natural.date}
    active=df[df.ramp>=.01]
    for pattern in ("synchronous","lagged","opposing"):
        pool=active[active[pattern]>=active[pattern].quantile(.75)]
        for label,q in (("tight",.15),("loose",.85)):
            avail=pool[~pool.date.isin(used)]
            i=(avail.device_headroom-pool.device_headroom.quantile(q)).abs().idxmin()
            row=avail.loc[i]
            selected.append(dict(**row.to_dict(),pattern=pattern,reserve=label))
            used.add(row.date)
    w=pd.DataFrame(selected)
    assert w.date.nunique()==7 and all(w.date.str.startswith("201005"))
    w.to_csv(OUT/"windows.csv",index=False)
    arrays={"fixed":fixed}
    for row in w.itertuples():
        for name,value in zip(("history","pv","initial","taps","slow"),data[row.slot]):
            arrays[row.date+"_"+name]=value
    np.savez_compressed(OUT/"windows.npz",**arrays)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/"lock.json").write_text(json.dumps(dict(april_sha=sha(ARCHIVE),may_sha=sha(MAY),predictors=hashes,
        history_candidate="April 7 0s AC-OPF first certified command, fixed before May replay",
        candidates=len(df),april_analogues=len(features),windows_sha=sha(OUT/"windows.csv"),
        fixed_inputs_sha=sha(OUT/"windows.npz"),dev_sha=sha(OUT/"development.npz")),indent=2))
    print(w[["date","pattern","reserve","device_headroom","voltage_headroom"]].to_string(index=False),flush=True)


if __name__=="__main__":
    select()
