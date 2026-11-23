"""Same causal release gate applied to A, B, C and their paired ablations."""
import itertools
import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from ieee123_chen_windows import OUT, ROOT, joint, build, phase_power
from ieee123_interval_ac import BatchAC, Snapshot
from ieee123_service_areas import metrics
from ieee123_forecast_value import FIELDS, MARGIN

ARMS=("A0","A","B","M","C")
SEEDS=(11,23,37)
TRAIN=OUT/"training"


class Expired(Exception):
    pass


def forecast_net(path):
    net=joint.Forecaster()
    net.load_state_dict(torch.load(path,weights_only=True,map_location="cpu")["fore"])
    return net.eval()


def make_gate(plant,old,fixed,past,path,begin):
    current=phase_power(past[-1])
    grid,weights=np.unique(phase_power(np.vstack((past[-1],path))),axis=0,return_counts=True)
    cold=Snapshot(plant.m,plant.slot,plant.taps,current)
    nv=2*plant.m.original_n
    events=[]
    cached={}
    best=None

    def eligible_margins(raw):
        return raw[:nv].min()>=0 and raw[-54:].min()>=0 and raw[nv:-54].min()>=1-(1+1e-7)**2

    def inspect(cmd,kind):
        nonlocal best
        if perf_counter()-begin>=5:
            raise Expired()
        if cmd is None:
            return
        cmd=np.asarray(cmd,dtype=float).reshape(54)
        key=cmd.tobytes()
        if not np.isfinite(cmd).all() or abs(cmd).max()>1 or key in cached:
            return
        cached[key]=True
        try:
            y=plant.evaluate(cmd,grid)
            raw=y[:,1:].min(0)
            if not eligible_margins(raw):
                return
            score=(max(0.,MARGIN-raw[:nv].min()),float(np.average(y[:,0],weights=weights)))
            if best is not None and not (score[0]<best[0]-1e-9 or
                    (abs(score[0]-best[0])<=1e-9 and score[1]<best[1]-1e-7)):
                return
            yy=cold.evaluate(cmd)[0][1:]
            if not eligible_margins(yy):
                return
            if min(yy.min(),raw.min())<0:
                for power in grid:
                    if perf_counter()-begin>=5:
                        raise Expired()
                    st=plant.m.solve(plant.slot,cmd,"high",plant.taps,generation_kw=power,tolerance=1e-10,max_iterations=1500)
                    m=metrics(plant.m,st,plant.slot,"high","release",plant.m.ratings.ravel())
                    if not (.95<=m["vmin"] and m["vmax"]<=1.05 and m["main_util"]<=1+1e-7 and m["port_util"]<=1):
                        return
        except ArithmeticError:
            return
        stamp=perf_counter()-begin
        if stamp>5:
            raise Expired()
        events.append((stamp,kind,cmd.copy()));best=score
    return events,inspect


@torch.inference_mode()
def issue(arm,plant,old,fixed,past,ref,fore,action):
    begin=perf_counter()
    h=joint.tensor(past[None])
    reference=ref(h).numpy()[0]
    events,inspect=make_gate(plant,old,fixed,past,reference,begin)
    path=None
    proposal=None
    proposal_time=np.nan
    fail=0
    try:
        inspect(old,0)
        inspect(fixed,1)
        path=fore(h).numpy()[0]
        t0=perf_counter()
        proposal=action(plant,old,fixed,past,path)
        proposal_time=perf_counter()-t0
        if proposal is not None and np.isfinite(proposal).all():
            inspect(proposal,2)
        else:
            fail=1
    except Expired:
        fail=2
    except ArithmeticError:
        fail=1
    return events,path,proposal,perf_counter()-begin,proposal_time,fail


def checkpoint(arm,seed):
    return joint.OUT/f"init_{seed}.pt" if arm=="A0" else TRAIN/f"{arm}_{seed}.pt"


def integrity():
    from ieee123_chen_actors import __file__ as actor_file
    from ieee123_chen_projection import __file__ as a_file
    paths=[Path(__file__),Path(actor_file),Path(a_file),OUT/"windows_lock.json",
           ROOT/"tools/ieee123_six.py",ROOT/"tools/ieee123_interval_ac.py",ROOT/"tools/ieee123_service_areas.py",
           ROOT/"configs/ieee123_service_areas.json",joint.OUT/"init_11.pt"]
    paths += [checkpoint(arm,seed) for arm,seed in itertools.product(ARMS,SEEDS)]
    for seed,arm in itertools.product(SEEDS,ARMS[1:]):
        ck=checkpoint(arm,seed)
        meta=json.loads(ck.with_suffix(".json").read_text())
        assert joint.sha(ck)==meta.get("selected_sha",meta.get("checkpoint_sha"))
    return {str(p.relative_to(ROOT)):joint.sha(p) for p in paths}


def verify_windows():
    lock=json.loads((OUT/"windows_lock.json").read_text())
    assert joint.sha(OUT/"windows.npz")==lock["windows"] and joint.sha(OUT/"windows.csv")==lock["manifest"]
    return pd.read_csv(OUT/"windows.csv",dtype={"date":str})


def run():
    from ieee123_chen_actors import load_policy
    from ieee123_chen_projection import optimize
    windows=verify_windows()
    hashes=integrity()
    lock=OUT/"eval_lock.json"
    if lock.exists():
        assert json.loads(lock.read_text())==hashes
    else:
        lock.write_text(json.dumps(hashes,indent=2))
    tag=joint.sha(lock)
    ref=forecast_net(joint.OUT/"init_11.pt")
    _,model,*_=build()
    keys=list(itertools.product(SEEDS,ARMS))
    nets={}
    for seed,arm in keys:
        actor=load_policy(arm,seed) if arm in ("B","C","M") else None
        nets[seed,arm]=(forecast_net(checkpoint(arm,seed)),
            (lambda p,o,f,h,s,net=actor: net(p,o,h,s)) if actor else optimize)
    dest=OUT/"replay";dest.mkdir(exist_ok=True)
    with np.load(OUT/"windows.npz") as archive:
        fixed=archive["fixed"]
        for j,row in enumerate(windows.itertuples()):
            path=dest/f"{row.date}.npz"
            if path.exists():
                with np.load(path) as old_file:
                    assert str(old_file["signature"])==tag
                continue
            tdate=perf_counter()
            seq=archive[row.date+"_pv"]
            truth=phase_power(seq[10:610])
            plant=BatchAC(model,row.slot,tuple(archive[row.date+"_taps"]))
            initial=archive[row.date+"_initial"]
            old={key:initial.copy() for key in keys}
            last_v={key:None for key in keys}
            arrays={}
            for seed,arm in keys:
                issue(arm,plant,initial,fixed,np.tile(seq[0],(11,1)),ref,*nets[seed,arm])
            for i in range(120):
                k=i*5
                past=seq[k:k+11]
                # Replay is outside online timing. Exact duplicates on the same
                # realized second solve the same deterministic fixed-tap circuit.
                realized={}
                order=(i+j)%len(keys)
                for key in keys[order:]+keys[:order]:
                    seed,arm=key
                    prefix=f"{arm}{seed}_"
                    def add(name,v):
                        arrays.setdefault(prefix+name,[]).append(v)
                    due,path5,proposal,elapsed,opt_s,fail=issue(arm,plant,old[key],fixed,past,ref,*nets[key])
                    add("start",old[key]);add("time",elapsed);add("opt_s",opt_s);add("fail",fail)
                    add("pred",path5 if path5 is not None else np.full((5,6),np.nan))
                    add("proposal",proposal if proposal is not None else np.full(54,np.nan))
                    for stamp,kind,cmd in due:
                        add("events",(k,stamp,kind));add("event_actions",cmd)
                    for s in range(5):
                        ready=[x for t,_,x in due if t<=s]
                        applied=ready[-1] if ready else old[key]
                        lookup=(s,applied.tobytes())
                        if lookup not in realized:
                            try:
                                st=model.solve(row.slot,applied,"high",plant.taps,generation_kw=truth[k+s],tolerance=1e-8,max_iterations=250)
                            except ArithmeticError:
                                st=model.solve(row.slot,applied,"high",plant.taps,generation_kw=truth[k+s],start=last_v[key],tolerance=1e-8,max_iterations=1500)
                            v=metrics(model,st,row.slot,"high",arm,model.ratings.ravel())
                            realized[lookup]=st["v"],[v[f] for f in FIELDS]
                        last_v[key],obs=realized[lookup]
                        add("actual",applied);add("meters",obs)
                    if due:old[key]=due[-1][2]
                    add("end",old[key])
            tmp=path.with_suffix(".part.npz")
            np.savez_compressed(tmp,signature=tag,**arrays)
            tmp.replace(path)
            print("completed",row.date,round(perf_counter()-tdate,1),flush=True)


if __name__=="__main__":
    torch.set_num_threads(1)
    with threadpool_limits(1):
        run()
