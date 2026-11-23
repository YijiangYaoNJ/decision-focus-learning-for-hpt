"""Compare three information sets with the existing rolling 54-dimensional scenario QP."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from ieee123_interval_run import dispatch, BatchAC, build, metrics
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from ieee123_joint5_trajectories import JUNE, OUT as PREV, verify_file, joint
from ieee123_interval_data import phase_power
from ieee123_ramp_windows import ROOT, SIX, day_data
from ieee123_preference_replay import MARGINS

OUT=ROOT/"runs/ieee123_five_second_information_v1"
ARMS=("persistence","network","oracle")
CKPT=PREV/"mse5_11.pt"  # first pre-existing training seed; no June-based model selection
FIELDS=("vmin","vmax","main_util","port_util","network_kw","heat_kw")
MARGIN=MARGINS[1]


def inputs():
    for name in ("manifest.csv","june.npz"):verify_file(name)
    source=json.loads((PREV/"lock.json").read_text())
    assert joint.sha(JUNE)==source["june_sha"]
    assert joint.sha(CKPT)==json.loads(CKPT.with_suffix(".json").read_text())["selected_sha"]
    w=pd.read_csv(PREV/"manifest.csv",dtype={"date":str}).query("split=='test'")
    assert w.date.nunique()==len(w)==10
    with np.load(PREV/"june.npz") as f: data=dict(f)
    with zipfile.ZipFile(JUNE) as z:
        for row in w.itertuples():
            raw=day_data(z,row.date+".txt")[row.second-10:row.second+601,SIX]
            assert raw.shape==(611,6) and np.isfinite(raw).all()
            np.testing.assert_array_equal(raw[:-1],data[row.date+"_pv"])
            data[row.date+"_pv"]=raw
    return w,data


def signature():
    files=(CKPT,PREV/"lock.json",Path(__file__),ROOT/"tools/ieee123_interval_run.py",
           ROOT/"tools/ieee123_interval_ac.py",ROOT/"tools/ieee123_ramp_dispatch.py")
    return {p.name:joint.sha(p) for p in files}


@torch.inference_mode()
def run():
    w,data=inputs()
    hashes=signature()
    protocol=dict(files=hashes, dates=w.date.tolist(), margin=MARGIN,controls=54,
                  cap_s=5, current_plus_future_seconds=6, normalized_material_delta=.01,
                  replay="actual release timestamps; old action until certified command arrives",
                  model="expanded-pool MSE forecast, first pre-existing seed 11; no fitting",
                  optimizer="existing eight-iteration shared-current-Jacobian scenario QP with nonlinear line search")
    lock=OUT/"protocol.json"
    if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError("Run protocol changed")
    lock.write_text(json.dumps(protocol,indent=2))
    tag=joint.sha(lock)
    net,_=joint.load_models(11,CKPT);net.eval()
    _,model,*_=build()
    for j,row in enumerate(w.itertuples()):
        dest=OUT/f"{row.date}.npz"
        if dest.exists():
            with np.load(dest) as cache:
                assert str(cache["signature"])==tag
            continue
        seq=data[row.date+"_pv"]; taps=tuple(data[row.date+"_taps"])
        plant=BatchAC(model,row.slot,taps);fixed=data["fixed"];initial=data[row.date+"_initial"]
        old={a:initial.copy() for a in ARMS}
        voltage={a:None for a in ARMS}
        arrays={a:{k:[] for k in ("start","final","actual","meters","forecast","events","event_actions",
                                    "latency","first_new","status","retry")} for a in ARMS}
        grid=phase_power(seq[10:])
        def issue(kind,past,truth,previous):
            used=[]
            def forecast():
                if kind=="persistence":path=np.tile(past[-1],(5,1))
                elif kind=="network":path=net(joint.tensor(past[None])).numpy()[0]
                else:path=truth.copy()
                used.append(path)
                return path
            releases,info=dispatch("scenario",plant,previous,fixed,past,None,None,margin=MARGIN,
                                   future_path=forecast)
            return releases,info,used[0]
        # Warm all code paths before the measured run using a constant pre-window observation.
        for a in ARMS:issue(a,np.tile(seq[0],(11,1)),np.tile(seq[0],(5,1)),initial)
        for cycle in range(120):
            k=5*cycle;past=seq[k:k+11];truth=seq[k+11:k+16]
            for shift in range(3):
                arm=ARMS[(cycle+j+shift)%3];r=arrays[arm]
                # Isolate future measurements from both causal controller input paths.
                due,info,used=issue(arm,past,truth if arm=="oracle" else None,old[arm])
                r["start"].append(old[arm]);r["forecast"].append(used);r["latency"].append(info["elapsed_s"])
                r["status"].append(info["status"]);r["first_new"].append(info["first_new_s"])
                for t,cmd,_ in due:
                    r["events"].append((k,t));r["event_actions"].append(cmd)
                for s in range(5):
                    ready=[cmd for t,cmd,_ in due if t<=s]
                    applied=ready[-1] if ready else old[arm]
                    retry=0
                    try:
                        st=model.solve(row.slot,applied,"high",taps,generation_kw=grid[k+s],tolerance=1e-8,max_iterations=250)
                    except ArithmeticError:
                        retry=1
                        st=model.solve(row.slot,applied,"high",taps,generation_kw=grid[k+s],start=voltage[arm],
                                       tolerance=1e-8,max_iterations=1500)
                    voltage[arm]=st["v"];obs=metrics(model,st,row.slot,"high",arm,model.ratings.ravel())
                    r["actual"].append(applied);r["meters"].append([obs[f] for f in FIELDS]);r["retry"].append(retry)
                if due:old[arm]=due[-1][1]
                r["final"].append(old[arm])
        serialized={f"{a}_{k}":np.asarray(v) for a,r in arrays.items() for k,v in r.items()}
        tmp=dest.with_suffix(".part.npz")
        np.savez_compressed(tmp,**serialized,signature=tag);tmp.replace(dest)
        print(row.date,"complete",flush=True)


if __name__=="__main__":
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    with threadpool_limits(1):
        if sys.argv[1:]==["run"]:run()
        elif sys.argv[1:]==["report"]:
            from ieee123_forecast_value_report import report
            report()
        else:raise SystemExit("run or report")
