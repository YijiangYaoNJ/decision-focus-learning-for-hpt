"""Preselect Oahu one-second weather windows independently of network results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/ieee123_fast_ramps_v1"
ARCHIVE = ROOT.parent / "networks/oahu_201004.zip"
# One-based columns in MIDC's raw GHI dictionary. Fixed proxy assignment to
# OEDI profile order 7,29,49,55,63,68,80,87,113.
FIELDS = [(5,6),(7,8),(9,10),(11,12),(14,16),(17,18),(19,20),(21,22),(22,23)]
SIX = [0,1,4,2,3,8]
SECONDS = 30


def day_data(z, name):
    x = pd.read_csv(z.open(name),header=None).to_numpy()
    hhmm = x[:,3].astype(int)
    ix = (hhmm//100*3600 + hhmm%100*60 + x[:,0]).astype(int)
    date = datetime.strptime(name[:8],"%Y%m%d")
    assert np.all(x[:,1]==date.year) and np.all(x[:,2]==date.timetuple().tm_yday)
    assert len(np.unique(ix))==len(ix) and np.all(np.diff(ix)>0)
    pv = np.full((86400,9),np.nan)
    for j,(i,k) in enumerate(FIELDS):
        a = x[:,[i-1,k-1]]
        good = np.all((a>=0)&(a<=1500),axis=1)
        pv[ix[good],j] = np.clip(a[good].mean(1)/1000,0,1)
    return pv


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    rows = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in sorted(z.namelist()):
            pv = day_data(z,name)
            # Keep the whole window inside one 15-minute load interval.
            for t in range(36000,50400,SECONDS):
                if t%900<10: continue
                seq = pv[t-5:t+SECONDS,SIX]
                if np.isfinite(seq).all():
                    score = np.sqrt(np.mean((seq[5:]-seq[:-5])**2))
                    rows.append((name,t,score))
        df = pd.DataFrame(rows,columns=["file","second","rms_5s_pu"])
        chosen,used,inputs = [],set(),{}
        for label,pct in [("steady",.1),("moderate",.5),("fast",.95)]:
            target = df.rms_5s_pu.quantile(pct)
            count = 0
            for row in df.iloc[np.argsort(abs(df.rms_5s_pu-target),kind="stable")].itertuples():
                if row.file in used: continue
                pv = day_data(z,row.file)
                t = row.second
                boundary = t//900*900
                past = pv[boundary-3600:boundary].reshape(4,900,9)
                if not np.isfinite(past).all(): continue
                key = row.file[:8]
                slot = (datetime.strptime(key,"%Y%m%d").timetuple().tm_yday-1)*96+t//900
                inputs[key+"_history"] = np.r_[past.mean(1)[::-1].ravel(),
                                             np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96)]
                inputs[key+"_pv"] = pv[t-5:t+SECONDS,SIX]
                chosen.append(dict(date=key,second_hst=t,slot=slot,band=label,
                                   rms_5s_pu=row.rms_5s_pu,target=target))
                used.add(row.file)
                count+=1
                if count==2: break
            assert count==2
    pd.DataFrame(chosen).to_csv(OUT/"windows.csv",index=False)
    np.savez(OUT/"local_inputs.npz",**inputs)
    (OUT/"selection.json").write_text(json.dumps(dict(archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        eligible_windows=len(rows),seconds=SECONDS,quantiles=[.1,.5,.95],fields=FIELDS,six_indices=SIX,
        source="https://midcdmz.nlr.gov/oahu_archive/rawdata/Oahu_GHI/201004.zip",
        field_dictionary="https://midcdmz.nlr.gov/apps/html.pl?site=oahugrid&page=raw_ghi"),indent=2))
    print(pd.DataFrame(chosen).to_string(index=False),flush=True)


if __name__=="__main__":
    select()
