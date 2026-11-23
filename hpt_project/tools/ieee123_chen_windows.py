"""Lock unused June dates before any three-arm test replay."""
import json
import re
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ieee123_joint5_trajectories import JUNE, OUT as APRIL, choose, verify_file, joint
from ieee123_fast_ramps import predictor
from ieee123_service_areas import build, local, ZERO
from ieee123_interval_data import phase_power

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/ieee123_chen_three_arm_v1"


def prepare():
    sys.path.insert(0,str(ROOT/"src"))
    verify_file("manifest.csv")
    verify_file("june.npz")
    old=json.loads((APRIL/"lock.json").read_text())
    assert joint.sha(JUNE)==old["june_sha"]
    used=set()
    records=[]
    for folder in sorted((ROOT/"runs").glob("ieee123*")):
        if folder==OUT:
            continue
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            dates=set(re.findall(r"(?<![0-9.])201006[0-3][0-9](?![0-9])",path.name))
            if path.suffix in (".csv",".json") and path.stat().st_size<16_000_000:
                dates.update(re.findall(r"(?<![0-9.])201006[0-3][0-9](?![0-9.])",path.read_text(encoding="utf-8-sig",errors="replace")))
            if dates:
                used.update(dates)
                records.append(dict(file=str(path.relative_to(ROOT)),dates=sorted(dates)))
    choices=choose(JUNE,1,np.random.default_rng(32471))
    held=[r for r in choices if str(r[0]) not in used]
    assert held and len(held)<=20 and len({r[0] for r in held})==len(held)
    slow,_=predictor(np.stack([r[3] for r in held]))
    _,m,*_=build()
    with np.load(APRIL/"june.npz") as z:
        data={"fixed":z["fixed"]}
    rows=[]
    for (date,t,slot,history,pv),pred in zip(held,slow):
        assert pv.shape==(611,6)
        st=m.step(slot,ZERO,"high",generation_kw=phase_power(pred))
        data.update({f"{date}_pv":pv,f"{date}_taps":st["taps"],f"{date}_initial":local(st,m,ZERO)})
        rows.append(dict(date=date,second=t,slot=slot))
    OUT.mkdir(exist_ok=True)
    if (OUT/"windows_lock.json").exists():
        raise FileExistsError("Evaluation windows are already locked.")
    np.savez_compressed(OUT/"windows.npz",**data)
    pd.DataFrame(rows).to_csv(OUT/"windows.csv",index=False)
    summary=dict(june=joint.sha(JUNE),old_dates=sorted(used),dates=[r["date"] for r in rows],files=records,
            windows=joint.sha(OUT/"windows.npz"),manifest=joint.sha(OUT/"windows.csv"),selection_seed=32471,
            primary="Actual network plus converter kWh at non-increased voltage and equipment risk; paired date C-B and C-A; no test tuning",
            seeds=[11,23,37],period_s=5,replay_s=1,window_s=600,reference_fore="joint.OUT/init_11.pt",
            common_gate="old/fixed/proposal; hard AC limits, 0.003 voltage margin then predicted mean loss",
            inference="one thread, elapsed includes forecast, proposal and release checks")
    (OUT/"windows_lock.json").write_text(json.dumps(summary,indent=2))
    print("Previously used",sorted(used),"locked",summary["dates"],flush=True)


if __name__=="__main__":
    prepare()
