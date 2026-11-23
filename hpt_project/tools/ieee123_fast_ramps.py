"""Run one-second causal replay. Each solver carries its own actually issued action."""
import os
for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[variable]="1"

import hashlib
import json
import platform
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
import osqp
import torch
from threadpoolctl import threadpool_limits

from ieee123_ramp_windows import OUT,ROOT,SIX,SECONDS
from ieee123_ramp_dispatch import solve,Snapshot
from ieee123_service_areas import build,local,metrics,ZERO

METHODS=("QP","SL-QP","AC-OPF")


def predictor(hist):
    # Restore the existing 123 forecaster and PV repair head only. No dispatcher.
    from cross_joint_train import Chain
    from feeder_joint_backend import Repair
    base=ROOT/"runs/cross_joint_v1/123"
    repair_path=ROOT/"runs/joint_feeders_v2/123/11_joint_mse_selected.pt"
    base_path=base/"11/initial.pt"
    with np.load(base/"ledger.npz") as ledger:
        fore=Chain(ledger)
    fore.load_state_dict(torch.load(base_path,weights_only=True))
    repair=Repair(38,9,9)
    repair.load_state_dict(torch.load(repair_path,weights_only=True)["pv"])
    h=torch.as_tensor(hist,dtype=torch.float32)
    with torch.inference_mode():
        p=repair.eval()(h,fore.eval().predict(h)).numpy()
    return p[:,SIX],{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()
                      for x in (base_path,repair_path)}


def run():
    windows=pd.read_csv(OUT/"windows.csv",dtype={"date":str})
    data=np.load(OUT/"local_inputs.npz")
    forecasts,hashes=predictor(np.stack([data[k+"_history"] for k in windows.date]))
    _,model,_,_,_,_=build()
    capacity=np.array([p["kva"] for p in model.cfg["high_pv"]])
    expand=lambda a:np.repeat(a*capacity/3,3,axis=-1)
    timings=[]; replay=[]; holds=[]; logs=[]; actions={}; warm=False
    meta=dict(forecast_sha256=hashes,python=platform.python_version(),platform=platform.platform(),
              scipy=scipy.__version__,osqp=osqp.__version__,cpu=os.environ.get("PROCESSOR_IDENTIFIER"),
              capacities_kw=capacity.tolist(),threads=1)
    (OUT/"protocol.json").write_text(json.dumps(meta,indent=2))
    for idx,row in enumerate(windows.itertuples()):
        pv=expand(data[row.date+"_pv"])
        slow=expand(forecasts[idx])
        slot=row.slot
        slowstate=model.step(slot,ZERO,"high",generation_kw=slow)
        taps=slowstate["taps"]
        init=local(slowstate,model,ZERO)
        # A shared pre-window dispatch five seconds before the first measured cycle.
        seed,pre=solve(model,slot,taps,pv[0],init,"SL-QP")
        initial=seed[-1][1] if seed else init
        if not warm:
            for name in METHODS:
                solve(model,slot,taps,pv[1],init,name)
            warm=True
        def observe(a,g):
            s=model.solve(slot,a,"high",taps,generation_kw=g,tolerance=1e-8,max_iterations=250)
            m=metrics(model,s,slot,"high","replay",model.ratings.ravel())
            return {k:float(m[k]) for k in ("vmin","vmax","excursion","main_util","port_util","loss_kw")}|{
                "violation":m["violation"],"overload":m["overload"]}
        y,s=Snapshot(model,slot,taps,pv[0]).evaluate(initial)
        m=observe(initial,pv[0])
        assert (min(y[1:])>=0)==(not (m["violation"] or m["overload"])) and abs(y[0]-m["loss_kw"])<1e-5
        common=dict(date=row.date,band=row.band)
        for i in range(SECONDS):
            holds.append(dict(**common,second=i,**observe(initial,pv[5+i])))
        state={name:initial.copy() for name in METHODS}
        actions[row.date+"_initial"]=initial
        actions[row.date+"_taps"]=taps
        actions[row.date+"_slow_kw"]=slow
        for offset in range(0,SECONDS,5):
            for mi in range(3):
                name=METHODS[(idx+offset//5+mi)%3]
                old=state[name]
                # Last measurement available at issue, held for five seconds.
                events,times=solve(model,slot,taps,pv[5+offset],old,name)
                key=f"{row.date}_{offset}_{name}"
                actions[key+"_old"]=old
                for j,(et,ea,loss) in enumerate(events):
                    actions[f"{key}_{j}"]=ea
                    logs.append(dict(**common,method=name,offset=offset,event=j,time_s=et,loss_kw=loss))
                due=[e for e in events if e[0]<=5]
                state[name]=due[-1][1] if due else old
                timings.append(dict(**common,offset=offset,method=name,**times,feasible_by_5=bool(due),
                    new_by_5=any(np.max(abs(a-old))>1e-8 for _,a,_ in due)))
                for sec in range(5):
                    delivered=[e for e in due if e[0]<=sec]
                    cmd=delivered[-1][1] if delivered else old
                    g=pv[5+offset+sec]
                    online,held=observe(cmd,g),observe(old,g)
                    instant=observe(state[name],g)
                    replay.append(dict(**common,method=name,offset=offset,second=sec,
                        **online,held_bad=held["violation"] or held["overload"],
                        held_voltage=held["violation"],held_overload=held["overload"],
                        immediate_bad=instant["violation"] or instant["overload"]))
            print(row.date,row.band,offset,[round(x["elapsed_s"],2) for x in timings[-3:]],flush=True)
        for path,entries in [("timing",timings),("replay",replay),("hold",holds),("incumbents",logs)]:
            pd.DataFrame(entries).to_csv(OUT/(path+".csv"),index=False)
        np.savez_compressed(OUT/"commands.npz",**actions)
    assert len(timings)==len(windows)*SECONDS//5*3 and len(replay)==len(timings)*5


if __name__=="__main__":
    with threadpool_limits(1):
        run()
