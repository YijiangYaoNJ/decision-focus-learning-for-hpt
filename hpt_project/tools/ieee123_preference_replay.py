"""Time-stamped closed-loop five-second dispatch on the locked service-area feeder."""
from ieee123_interval_run import dispatch, BatchAC, build, metrics, phase_power, PRIOR
from ieee123_interval_data import OUT as SOURCE, feature
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit
from threadpoolctl import threadpool_limits

OUT=SOURCE.with_name("ieee123_service_preference_v1")
MARGINS=(0.,.003,.006)
DEADBANDS=(.035,.020,.005)
SEEDS=(11,23,37)
CLASSIC=("droop","sensitivity","economic_qp","snapshot","margin_opf","scenario")
LEARNED=("imitation","imitation_mse","original","preference")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fixed_data():
    lock=json.loads((SOURCE/"lock.json").read_text())
    for name,key in (("windows.csv","windows_sha"),("windows.npz","fixed_inputs_sha"),("development.npz","dev_sha")):
        assert sha(SOURCE/name)==lock[key]
    return pd.read_csv(SOURCE/"windows.csv",dtype={"date":str}),np.load(SOURCE/"windows.npz")

def features(m,slot,taps,past):
    load=m.series[slot].sum()
    return np.r_[feature(past),load.real/1000,load.imag/1000,np.asarray(taps)/16,
                 np.sin(2*np.pi*slot/96),np.cos(2*np.pi*slot/96)]

def policy(weights,x,mode):
    r=mode/2
    h=np.r_[(x-weights["mean"])/weights["scale"],r]
    for j in (0,2):
        h=weights[f"body.{j}.weight"]@h+weights[f"body.{j}.bias"]
        h=h*expit(h)
    a=np.tanh(weights["action.weight"]@np.r_[h,h*r]+weights["action.bias"]).reshape(18,3)
    a[:,:2]/=np.maximum(1,np.linalg.norm(a[:,:2],axis=1,keepdims=True))
    return a.ravel()

def run(methods,modes=(0,1,2),dates=None,*,policy_loader=None,output=OUT,case_data=None):
    windows,data=fixed_data() if case_data is None else case_data()
    if dates is not None:
        if not set(dates)<=set(windows.date):raise ValueError("Date outside locked May windows")
        windows=windows[windows.date.isin(dates)]
    dev=np.load(SOURCE/"development.npz")
    tree,library=cKDTree(dev["features"]),dev["futures"]
    _,model,*_=build()
    for method in methods:
        for mode in (r for r in modes if method!="snapshot" or r==0):
            for seed in (SEEDS if method in LEARNED or policy_loader else (0,)):
                folder=output/"replay"/f"{method}_{mode}_{seed}"
                folder.mkdir(parents=True,exist_ok=True)
                weights=None
                external=policy_loader(method,seed) if policy_loader else None
                if method in LEARNED:
                    path=OUT/f"{method}_{seed}.npz"
                    lock=json.loads((OUT/f"{method}_{seed}.json").read_text())
                    assert sha(path)==lock["weights_sha256"]
                    with np.load(path) as z: weights=dict(z)
                for row in windows.itertuples():
                    dest=folder/f"{row.date}.npz"
                    if dest.exists():
                        if external is None:continue
                        with np.load(dest) as saved:
                            if str(saved.get("checkpoint_sha",""))==external.checkpoint_sha:continue
                    reference=OUT/"replay/snapshot_0_0"/dest.name
                    if method=="margin_opf" and mode==0 and reference.is_file():
                        shutil.copyfile(reference,dest)
                        continue
                    d=row.date
                    taps=tuple(data[d+"_taps"])
                    plant=BatchAC(model,row.slot,taps)
                    seq=data[d+"_pv"]
                    grid=phase_power(seq)
                    fixed=data["fixed"]
                    old=data[d+"_initial"].copy()
                    arm="snapshot" if method=="margin_opf" else method
                    def issue(old,past):
                        def propose():
                            feat=features(model,row.slot,taps,past)
                            return external(past,feat,mode) if external else policy(weights,feat,mode)
                        return dispatch(arm,plant,old,fixed,past,tree,library,margin=MARGINS[mode],
                               proposal=propose if weights is not None or external else None,details=True,deadband=DEADBANDS[mode])
                    issue(old,np.tile(seq[0],(11,1)))
                    elapsed=[]; first=[]; candidate=[]; selected=[]; samples=[]; cmds=[]; events=[]; ids=[]
                    last_voltage=None
                    replay_retries=0
                    for offset in range(0,600,5):
                        due,times=issue(old,seq[offset:offset+PRIOR+1])
                        elapsed.append(times["elapsed_s"]);first.append(times["first_s"])
                        candidate.append(times["proposal_ok"]);selected.append(times["proposal_selected"])
                        prior_id=len(cmds);cmds.append(old)
                        releases=[]
                        for t,a,_ in due:
                            assert 0<t<=5
                            releases.append((t,len(cmds)));events.append((offset,t,len(cmds)));cmds.append(a)
                        for sec in range(5):
                            ready=[k for t,k in releases if t<=sec]
                            k=ready[-1] if ready else prior_id
                            ids.append(k)
                            actual_grid=grid[offset+PRIOR+sec]
                            try:
                                state=model.solve(row.slot,cmds[k],"high",taps,generation_kw=actual_grid,
                                                  tolerance=1e-8,max_iterations=250)
                            except ArithmeticError:
                                replay_retries+=1
                                state=model.solve(row.slot,cmds[k],"high",taps,generation_kw=actual_grid,
                                                  start=last_voltage,tolerance=1e-8,max_iterations=1500)
                            last_voltage=state["v"]
                            obs=metrics(model,state,row.slot,"high",arm,model.ratings.ravel())
                            samples.append([obs[f] for f in ("vmin","vmax","main_util","port_util","network_kw","heat_kw")])
                        if due:old=due[-1][1]
                    tmp=dest.with_suffix(".part.npz")
                    np.savez_compressed(tmp,samples=samples,elapsed=elapsed,first=first,actions=cmds,index=ids,
                                        events=np.asarray(events).reshape(-1,3),candidate=candidate,selected=selected,
                                        replay_retries=replay_retries,
                                        **({"checkpoint_sha":external.checkpoint_sha} if external else {}))
                    tmp.replace(dest)
                    print(method,mode,seed,d,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("methods",nargs="+",choices=CLASSIC+LEARNED+("history",))
    parser.add_argument("--modes",nargs="+",type=int,choices=range(3),default=[0,1,2])
    parser.add_argument("--dates",nargs="+")
    args=parser.parse_args()
    with threadpool_limits(1):run(args.methods,args.modes,args.dates)
