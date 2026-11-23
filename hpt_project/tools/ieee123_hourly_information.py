"""Natural one-hour trajectories, causal QP comparisons and same-state action attribution."""
import json
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import ieee123_timing_training as study
from ieee123_ramp_windows import ARCHIVE, SIX, day_data
from ieee123_service_areas import local, ZERO
from ieee123_fast_ramps import predictor

base=study.base
OUT=study.OUT.with_name("ieee123_hourly_information_v1")
SECONDS=3600
MODES=("persistence","forecast","oracle","joint11","joint23","joint37")


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/"lock.json").exists():
        return
    rng=np.random.default_rng(211127)
    rows=[]; arrays={}; hashes={}; coverage=[]
    model=base.plant_model()
    sys.path.insert(0,str(OUT.parents[1]/"src"))
    for month in range(5,9):
        archive=ARCHIVE.with_name(f"oahu_2010{month:02}.zip")
        hashes[archive.name]=base.joint.sha(archive)
        with zipfile.ZipFile(archive) as source:
            eligible={}
            for file in sorted(source.namelist()):
                date=int(file[:8]); pv=day_data(source,file)
                for t in range(8*3600,15*3600+1,900):
                    if np.isfinite(pv[t-3600:t]).all() and np.isfinite(pv[t-10:t+SECONDS+1,SIX]).all():
                        eligible.setdefault(date,[]).append((file,t))
                coverage.append((date,len(eligible.get(date,[]))))
            for block in np.array_split(sorted(eligible),3):
                date=int(rng.choice(block))
                file,t=eligible[date][rng.integers(len(eligible[date]))]
                pv=day_data(source,file)
                slot=(datetime.strptime(str(date),"%Y%m%d").timetuple().tm_yday-1)*96+t//900
                h=pv[t-3600:t].reshape(4,900,9).mean(1)[::-1].ravel()
                angle=2*np.pi*slot/96
                pred,_=predictor(np.r_[h,np.sin(angle),np.cos(angle)][None,:])
                state=model.step(slot,ZERO,"high",generation_kw=base.joint.phase_power(pred[0]))
                arrays[f"{date}_initial"]=local(state,model,ZERO)
                arrays[f"{date}_taps"]=state["taps"]
                arrays[f"{date}_pv"]=pv[t-10:t+SECONDS+1,SIX]
                rows.append((date,t,slot))
    pd.DataFrame(coverage,columns=["date","eligible_starts"]).to_csv(OUT/"coverage.csv",index=False)
    pd.DataFrame(rows,columns=["date","second","slot"]).to_csv(OUT/"windows.csv",index=False)
    np.savez_compressed(OUT/"windows.npz",**arrays)
    protocol=dict(seed=211127,source=hashes,seconds=SECONDS,methods=MODES,dates=[r[0] for r in rows],
        population="May-August archives, first start 08:00, last 15:00, every 15 min; one random eligible day per chronological third per month and one random eligible start per selected day. Previously used months, developmental only.",
        state="Load updates every 15 minutes. Taps fixed at initialization. Carry all 54 executed settings across slot boundaries. One common frozen forecast in release screen.",
        tags="At common persistence state: spatial spread of realized normalized site ramps at +5s >=0.05; tight global minimum device headroom <=0.05 or voltage headroom <=0.005 pu. Separate device disparity: spread of six per-device worst port utilizations >=0.10. Ex-post strata only.",
        diagnostic="Every 60 s, compare three QP forecasts from exact persistence state. One-second common delay, same screen, actual five-second replay; retain proposed and effective device/phase deltas. Primary outcomes independent hour trajectories.",
        checkpoints="For each seed use end of full pass 2 aligned_joint (quarterly row 8); predetermined, unchanged from April comparison. No September data.",
        files={n:base.joint.sha(OUT/n) for n in ("windows.csv","windows.npz","coverage.csv")})
    (OUT/"lock.json").write_text(json.dumps(protocol,indent=2))


class GivenPath:
    def __init__(self,path):
        self.path=path
    def __call__(self,h):
        return h.new_tensor(self.path[None])


def action(mode,actor,fore,ref,plant,old,h,future):
    if mode.startswith("joint"):
        return base.issue(mode,actor,fore,ref,plant,old,h)
    path=ref
    if mode=="oracle":
        path=GivenPath(future)
    elif mode=="persistence":
        path=GivenPath(np.repeat(h[-1:],5,axis=0))
    return base.issue("qp",None,path,ref,plant,old,h)


def measure(model,plant,a,power,mode):
    try:
        state=model.solve(plant.slot,a,"high",plant.taps,generation_kw=power,tolerance=1e-8,max_iterations=250)
    except ArithmeticError:
        state=model.solve(plant.slot,a,"high",plant.taps,generation_kw=power,tolerance=1e-8,max_iterations=1500)
    return state,base.prior.metrics(model,state,plant.slot,"high",mode,model.ratings.ravel())


def scores(model,plant,old,new,delay,power,mode):
    ms=[measure(model,plant,new if new is not None and s>=delay else old,p,mode)[1] for s,p in enumerate(power)]
    return dict(kwh=sum(m["loss_kw"] for m in ms)/3600,voltage_s=sum(m["violation"] for m in ms),
                overload_s=sum(m["overload"] for m in ms),v_peak=max(m["excursion"] for m in ms),
                overload_peak=max(0.,max(max(m["main_util"],m["port_util"]) for m in ms)-1))


def run(job):
    date,mode=job
    path=OUT/f"{date}_{mode}.csv"
    if path.exists(): return
    model=base.plant_model()
    slot0=int(pd.read_csv(OUT/"windows.csv").set_index("date").loc[date,"slot"])
    with np.load(OUT/"windows.npz") as data:
        pv,old,taps=[data[f"{date}_{n}"] for n in ("pv","initial","taps")]
    ref=base.direct.replay.forecast_net(base.direct.windows.REF)
    actor=None; fore=ref
    if mode.startswith("joint"):
        seed=int(mode[5:])
        step=list(pd.read_csv(study.OUT/f"aligned_joint_{seed}"/"validation.csv").query("quarterly").step)[8]
        ck=study.OUT/f"{study.tag('aligned_joint',step)}_{seed}.pt"
        weights=torch.load(ck,weights_only=True)
        actor,fore=base.networks(seed,model)
        actor.load_state_dict(weights["actor"]);fore.load_state_dict(weights["fore"])
        actor.eval();fore.eval()
        (OUT/f"{mode}.sha.{date}").write_text(base.joint.sha(ck))
    rows=[]; diagnostics=[]; controls=[]; deltas=[]; pre_ports=[]
    powers=base.joint.phase_power(pv)
    for t in range(0,SECONDS,5):
        if t%900==0:
            plant=base.joint.BatchAC(model,slot0+t//900,tuple(taps))
        h=pv[t:t+11];future=pv[t+11:t+16]
        if t==0: action(mode,actor,fore,ref,plant,old,h,future)
        cmd,status,sec,_=action(mode,actor,fore,ref,plant,old,h,future)
        kw=scores(model,plant,old,cmd,sec,powers[t+10:t+15],mode)
        row=dict(date=date,mode=mode,t=t,ms=sec*1000,status=status,**kw)
        if mode=="persistence":
            state,m=measure(model,plant,old,powers[t+10],mode)
            util=np.asarray(state["ports"]).max(axis=0).reshape(6,3)
            pre_ports.append(1-util)
            row.update(spread=np.ptp(future[-1]-h[-1]),v_head=min(m["vmin"]-.95,1.05-m["vmax"]),
                       s_head=1-max(m["main_util"],m["port_util"]),port_spread=np.ptp(util.max(axis=1)))
            if t%60==0:
                for alt in MODES[:3]:
                    trace=[]
                    pre=ref if alt=="forecast" else GivenPath(future if alt=="oracle" else np.repeat(h[-1:],5,axis=0))
                    c,flag,_,_=base.issue("qp",None,pre,ref,plant,old,h,trace=trace)
                    executed=old if c is None else c
                    if alt=="persistence": ref_effective=executed.copy();ref_proposal=trace[0][1].copy()
                    deltas.append([trace[0][1]-ref_proposal,executed-ref_effective])
                    diagnostics.append(dict(date=date,t=t,mode=alt,status=flag,
                        **scores(model,plant,old,c,1,powers[t+10:t+15],alt)))
        rows.append(row)
        if cmd is not None: old=cmd
        controls.append(old.copy())
    if diagnostics:
        pd.DataFrame(diagnostics).to_csv(OUT/f"paired_{date}.csv",index=False)
        np.savez_compressed(OUT/f"paired_{date}.npz",delta=deltas,port_headroom=pre_ports)
    np.savez_compressed(path.with_suffix(".npz"),issued_or_held=controls)
    pd.DataFrame(rows).to_csv(path.with_suffix(".part"),index=False)
    path.with_suffix(".part").replace(path)
    print("done",date,mode,flush=True)


def report(lock):
    dfs=[]
    for date in lock["dates"]:
        tags=pd.read_csv(OUT/f"{date}_persistence.csv").set_index("t")
        labels=pd.DataFrame(dict(asynchronous=tags.spread>=.05,tight=(tags.s_head<=.05)|(tags.v_head<=.005),
                                unequal_ports=tags.port_spread>=.10))
        for mode in MODES:
            d=pd.read_csv(OUT/f"{date}_{mode}.csv").set_index("t")
            np.testing.assert_array_equal(d.index,np.arange(0,SECONDS,5))
            dfs.append(d.join(labels).reset_index())
    d=pd.concat(dfs,ignore_index=True)
    def aggregate(g):
        return g.groupby(["date","mode"],as_index=False).agg(kwh=("kwh","sum"),voltage_s=("voltage_s","sum"),
            overload_s=("overload_s","sum"),v_peak=("v_peak","max"),overload_peak=("overload_peak","max"),
            ms=("ms","median")).replace({"mode":{m:"joint" for m in MODES[3:]}}).groupby(["date","mode"]).mean()
    daily=aggregate(d)
    daily.to_csv(OUT/"by_date.csv")
    daily.groupby(level="mode").mean().to_csv(OUT/"summary.csv")
    groups=["asynchronous","tight","unequal_ports","mode"]
    d.groupby(groups).agg(n=("t","size"),wh_per_cycle=("kwh",lambda x:1000*x.mean()),
        voltage_s_per_cycle=("voltage_s","mean"),overload_s_per_cycle=("overload_s","mean")).to_csv(OUT/"strata.csv")
    out=[]
    for date in lock["dates"]:
        frame=pd.read_csv(OUT/f"paired_{date}.csv")
        delta=np.load(OUT/f"paired_{date}.npz")["delta"].reshape(-1,3,2,6,3,3)
        assert delta.shape==(60,3,2,6,3,3)
        for j,alt in enumerate(MODES[1:3],1):
            a=frame[frame["mode"]==alt].set_index("t"); b=frame[frame["mode"]=="persistence"].set_index("t")
            for k,t in enumerate(a.index):
                v=delta[k,j,1];dev,phase=np.unravel_index(np.argmax(abs(v).max(axis=2)),(6,3))
                out.append(dict(date=date,t=t,mode=alt,proposal_delta=np.max(abs(delta[k,j,0])),
                    command_delta=np.max(abs(v)),largest_device=int(dev+1) if np.any(v) else None,
                    phase="ABC"[phase] if np.any(v) else None,wh=1000*(a.loc[t,"kwh"]-b.loc[t,"kwh"]),
                    voltage_s=a.loc[t,"voltage_s"]-b.loc[t,"voltage_s"],overload_s=a.loc[t,"overload_s"]-b.loc[t,"overload_s"]))
    pd.DataFrame(out).to_csv(OUT/"same_state.csv",index=False)


if __name__=="__main__":
    if sys.argv[1]=="prepare": prepare()
    else:
        lock=json.loads((OUT/"lock.json").read_text())
        assert all(base.joint.sha(OUT/n)==sha for n,sha in lock["files"].items())
        if sys.argv[1]=="run": study.dev.run_pool(run,[(d,m) for m in MODES for d in lock["dates"]],18)
        report(lock)
