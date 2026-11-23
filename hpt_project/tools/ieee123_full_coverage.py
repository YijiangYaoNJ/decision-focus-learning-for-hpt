"""Full training-pool passes for the paired IEEE123 proposal-feedback experiment."""
import json
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import ieee123_proposal_train as short
from ieee123_proposal_windows import WEIGHTS, SEEDS

base=short.base
OUT=short.OUT.with_name("ieee123_full_coverage_v1")
MIN_PASSES=2
BATCH=8


def prepare():
    from ieee123_direct_confirm_data import previous
    sys.path.insert(0,str(previous.ROOT/"src"))
    OUT.mkdir(exist_ok=True)
    if (OUT/"lock.json").exists():
        return
    archive=previous.ARCHIVE.with_name("oahu_201007.zip")
    used=set()
    for path in previous.ROOT.joinpath("runs").glob("ieee123*/**/*.csv"):
        if OUT in path.parents or path.stat().st_size>64_000_000:
            continue
        if "date" in pd.read_csv(path,nrows=0).columns:
            col=pd.read_csv(path,usecols=["date"]).date.astype(str).str.replace("-","",regex=False)
            used.update(int(s) for s in col.unique() if len(s)==8 and s.isdigit() and s.startswith("201007"))
    draws=[r for r in previous.choose(archive,1,np.random.default_rng(211126)) if r[0] not in used]
    if not draws: raise ValueError("No eligible July dates")
    p,_=previous.predictor(np.stack([r[3] for r in draws]))
    model=base.plant_model()
    rows=[];arrays={}
    for (day,t,slot,_,pv),forecast in zip(draws,p):
        st=model.step(slot,previous.ZERO,"high",generation_kw=previous.phase_power(forecast))
        arrays.update({f"{day}_pv":pv,f"{day}_initial":previous.local(st,model,previous.ZERO),f"{day}_taps":st["taps"]})
        rows.append((day,t,slot))
    pd.DataFrame(rows,columns=["date","second","slot"]).to_csv(OUT/"windows.csv",index=False)
    np.savez_compressed(OUT/"windows.npz",**arrays)
    lock=dict(dates=[r[0] for r in rows],excluded_july=sorted(used),seeds=SEEDS,weights=WEIGHTS,
              manifest=base.joint.sha(OUT/"windows.csv"),windows=base.joint.sha(OUT/"windows.npz"),
              archive=base.joint.sha(archive),training_cases=5254,validation_cases=36,
              train="All 43 x 120 contiguous cycles plus 94 standalone states, each exactly once per pass. Eight states per update (last batch may be smaller). Same paired order and perturbations per seed.",
              stop="At least two full passes and four consecutive quarterly checks without a material improvement for either selection rule. No fixed maximum; epoch-boundary stop. Trade tolerance 1e-6 cost, strict 1e-6 risk or 1e-4 kW for a risk tie.",
              primary="B-A and C-B within preference, selectors reported separately, date paired. No use of July scores to change training.")
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2))
    print("Locked",len(rows),"July dates",flush=True)


def schedule(data,seed,epoch):
    rng=np.random.default_rng([seed,epoch,120])
    seq={int(s):list(range(s,s+120)) for s in data.starts}
    ids=set(data.train)-{i for block in seq.values() for i in block}
    blocks=list(seq.values())+[[int(i)] for i in sorted(ids)]
    order=list(rng.permutation(len(blocks)))
    active=[]
    while order or active:
        while order and len(active)<BATCH:
            active.append(iter(blocks[order.pop()]))
        batch=[];alive=[]
        for stream in active:
            i=next(stream,None)
            if i is not None: batch.append(i);alive.append(stream)
        active=alive
        if batch: yield batch


def atomic_save(value,path):
    tmp=path.with_suffix(".part")
    torch.save(value,tmp)
    tmp.replace(path)


def validate(actor,fore,data):
    with torch.no_grad():
        kw,v,vp,o,op,accept=base.validation(actor,fore,data)
    return kw,np.array([v,vp,o,op]),accept


class ProposalPowerFlow:
    """Retry difficult, unexecuted proposals using the same AC residual tolerance."""
    def __init__(self, plant):
        self.plant=plant
        self.m=plant.m

    def evaluate(self, action, power, *, max_iterations=250):
        try:
            return self.plant.evaluate(action,power,max_iterations=max_iterations)
        except ArithmeticError:
            if max_iterations<1500:
                raise
        for relaxation in (.5,.25,.1):
            try:
                return self.plant.evaluate(action,power,max_iterations=1500,relaxation=relaxation)
            except ArithmeticError:
                continue
        raise ArithmeticError("Proposal AC residual did not reach 1e-8 with damping")


def sample(data,i,last,actor,fore,rng,arm,w,*,forecast_feedback=True,applied_interval=False,final_cycle=None,last_frame_plant=None):
    plant,h,weather,truth=base.inputs(data,i)
    path=fore(short.tensor(h[None]))
    if not forecast_feedback:
        path=path.detach()
    # Use the same measured device features for the B-arm detached branch.
    load=plant.m.series[plant.slot]
    x=np.r_[short.prior.features(plant.m,plant.slot,plant.taps,h),
            np.column_stack((load.real,load.imag)).ravel()/100]
    obs=short.tensor(x[None]);hist=short.tensor(h[None])
    local=short.tensor(short.prior.local_features(plant,last,h)[None])
    cmd=actor(obs,hist,path,local)
    raw=short.prior.disk(cmd.detach().numpy())[0]
    grad,end,cost=execution_gradient(data,i,last,raw,rng,w,plant,weather,truth,
                                      applied_interval,final_cycle,last_frame_plant)
    signal=(cmd[0]*short.tensor(grad)).sum()
    if arm!="A":
        pg,_=short.proposal_gradient(ProposalPowerFlow(plant),raw,truth[1:])
        routed=cmd if arm=="C" else actor(obs,hist,path.detach(),local)
        signal=signal+(routed[0]*short.tensor(w*pg)).sum()
    return signal,end,cost


def execution_gradient(data,i,last,raw,rng,w,plant,weather,truth,applied_interval,final_cycle,last_frame_plant):
    dirs=rng.choice([-1.,1.],(short.prior.DIRECTIONS,54))
    delta=dirs*short.prior.EPSILON
    bank=short.prior.disk(np.vstack((raw,raw+delta,raw-delta)))
    execution_truth=truth
    if applied_interval:
        execution_truth=base.joint.phase_power(data.d["target"][i])
        if final_cycle is None:
            final_cycle=i >= data.starts[0] and (i-data.starts[0]+1)%120 == 0
        if final_cycle:
            execution_truth=execution_truth[:4]
    end,kw,risk,_=base.direct.outcomes(plant,last,bank,weather,execution_truth,first_second_held=not applied_interval,
                                      last_frame_plant=last_frame_plant)
    values=kw/100+w*(risk@short.RISK)
    grad=(values[1:1+len(dirs)]-values[1+len(dirs):])@dirs/(2*short.prior.EPSILON*len(dirs))
    return grad,end[0],values[0]


def train(arm,pref,seed,*,directory=None,observer=None):
    tag=f"{arm}_{pref}_{seed}"
    directory=OUT if directory is None else directory
    folder=directory/tag
    folder.mkdir(exist_ok=True)
    if (folder/"complete.json").exists(): return
    data=base.study_data()
    net,fore=base.networks(seed,data.model)
    opt=torch.optim.Adam([{"params":net.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    n=len(data.train); w=WEIGHTS[pref]
    step=0; epoch=0; best={}; material={}; stale=0; hist=[]
    resume=folder/"resume.pt"
    if resume.exists():
        r=torch.load(resume,weights_only=False)
        net.load_state_dict(r["net"]);fore.load_state_dict(r["fore"]);opt.load_state_dict(r["opt"])
        epoch,step,best,material,stale,hist,caps=(r[k] for k in ("epoch","step","best","material","stale","hist","caps"))
    def check(exposures,train_cost):
        nonlocal stale,caps
        kw,risks,accept=validate(net,fore,data)
        if not hist: caps=risks.copy()
        excess=float(np.maximum(0.,risks-caps-1e-8)@[1,50,2,20])
        trade=float(kw/100+w*(risks@short.RISK))
        improved=False
        for name,key in [("trade",(trade,)),("strict",(excess,float(kw)))]:
            if name not in best or key<best[name][0]:
                best[name]=(key,step)
                atomic_save(dict(actor=net.state_dict(),fore=fore.state_dict(),selected=step),directory/f"{arm}_{pref}_{name}_{seed}.pt")
            old=material.get(name)
            gain=(old is None or key[0]<old[0]-1e-6 or
                  (name=="strict" and abs(key[0]-old[0])<1e-10 and key[1]<old[1]-1e-4))
            if gain: material[name]=key;improved=True
        stale=0 if improved else stale+1
        hist.append(dict(step=step,exposures=exposures,kw=kw,vf=risks[0],vp=risks[1],of=risks[2],op=risks[3],
                         accepted=accept,trade=trade,excess=excess,train_exec_cost=train_cost,stale=stale))
        pd.DataFrame(hist).to_csv(folder/"validation.csv",index=False)
    if not hist:
        caps=None
        check(0,None)
        if observer is not None:
            observer(step,net,fore,True)
    setup=set(int(i) for i in data.starts)
    while True:
        epoch+=1; carried={}; sequence=[]; total=0.; checkpoint=1
        rng=np.random.default_rng([seed,epoch,54])
        batches=list(schedule(data,seed,epoch))
        for ids in batches:
            terms=[]
            for i in ids:
                last=carried.pop(i,data.d["old"][i])
                signal,end,cost=sample(data,i,last,net,fore,rng,arm,w)
                if i>=data.starts[0] and (i-data.starts[0]+1)%120: carried[i+1]=end
                terms.append(signal);total+=cost
            ix=rng.choice(data.train,128,replace=False)
            mse=(fore(short.tensor(data.d["past"][ix]))-short.tensor(data.d["target"][ix])).square().mean()
            opt.zero_grad()
            (torch.stack(terms).mean()+10*mse).backward()
            for mod in (net,fore): torch.nn.utils.clip_grad_norm_(mod.parameters(),3.,error_if_nonfinite=True)
            opt.step();step+=1;sequence.extend(ids)
            quarterly=4*len(sequence)>=checkpoint*n
            if quarterly:
                check((epoch-1)*n+len(sequence),total/len(sequence))
                checkpoint+=1
            if observer is not None:
                observer(step,net,fore,quarterly)
        assert not carried and sorted(sequence)==sorted(data.train.tolist()) and checkpoint==5
        np.save(folder/f"order_{epoch}.npy",np.asarray(sequence,dtype=np.int32))
        atomic_save(dict(net=net.state_dict(),fore=fore.state_dict(),opt=opt.state_dict(),epoch=epoch,
                         step=step,best=best,material=material,stale=stale,hist=hist,caps=caps),resume)
        if epoch>=MIN_PASSES and stale>=4:
            result=dict(epoch=epoch,updates=step,unique=n,exposures=epoch*n,seed=seed,
                        best_steps={k:v[1] for k,v in best.items()},stale=stale,
                        stop="full coverage and validation plateau")
            tmp=folder/"complete.part";tmp.write_text(json.dumps(result,indent=2));tmp.replace(folder/"complete.json")
            return


if __name__=="__main__":
    if sys.argv[1]=="prepare": prepare()
    else:
        torch.set_num_threads(1)
        with threadpool_limits(1):
            train(sys.argv[1],sys.argv[2],int(sys.argv[3]))
