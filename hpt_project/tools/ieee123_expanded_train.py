"""Paired CUDA training on the locked six-HPT continuous trajectories."""
import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import run
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import ieee123_expanded_data as data

base=data.base
full=data.hourly.study.full
OUT=Path(os.environ.get("HPT_TRAIN_OUTPUT", Path(__file__).resolve().parents[2] / "work" / "retrain")).resolve()
ARMS=("frozen","mse","joint")
SEEDS=(11,23,37)
BATCH=8
WEIGHT=4.


def record(path, value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2))
    temp.replace(path)


def paired_batches(dataset,seed,epoch):
    units=iter(dataset.epoch(seed,epoch))
    active=[]
    while True:
        while len(active)<BATCH:
            item=next(units,None)
            if item is None: break
            obj,indices,last=item
            active.append([obj,iter(indices),last])
        batch=[]
        for stream in active:
            index=next(stream[1],None)
            if index is not None: batch.append((stream,index))
        if not active: return
        active=[s for s,_ in batch]
        if batch: yield batch


def fit_epoch(ds,actor,fore,opt,arm,seed,epoch,device,folder):
    def tensor(array):
        return torch.as_tensor(np.asarray(array),dtype=torch.float32,device=device)
    directions=np.random.default_rng([seed,epoch,54])
    pools=[(ds.legacy.d["past"][ds.legacy.train],ds.legacy.d["target"][ds.legacy.train])]
    with np.load(data.OUT/"train.npz") as z:
        for key,r in ds.rows.iterrows():
            idx=data.make_index(z[key+"_pv"],int(r.slot))
            pools.append((idx["past"],idx["target"]))
    history,targets=[tensor(np.concatenate(v)) for v in zip(*pools)]
    exposures=len(history)
    draw=np.random.default_rng([seed,epoch,128])
    visited=updates=0
    total=0.
    start=perf_counter()
    for batch in paired_batches(ds,seed,epoch):
        rows=[]; obs=[]; local=[]; histories=[]
        for stream,i in batch:
            obj,_,last=stream
            p,h,weather,truth=base.inputs(obj,i)
            load=p.m.series[p.slot]
            obs.append(np.r_[base.prior.features(p.m,p.slot,p.taps,h),np.column_stack((load.real,load.imag)).ravel()/100])
            local.append(base.prior.local_features(p,last,h))
            histories.append(h)
            rows.append((stream,i,p,weather,truth))
        h=tensor(histories)
        path=fore(h)
        cmd=actor(tensor(obs),h,path if arm=="joint" else path.detach(),tensor(local))
        raw=base.prior.disk(cmd.detach().cpu().numpy())
        gradients=[]
        for action,(stream,i,p,weather,truth) in zip(raw,rows):
            obj,_,last=stream
            final=next_plant=None
            if isinstance(obj,data.Trajectory):
                final=i==data.CYCLES-1
                if not final and obj.d["slot"][i]!=obj.d["slot"][i+1]: next_plant=obj.plant(i+1)
            grad,stream[2],cost=full.execution_gradient(obj,i,last,action,directions,WEIGHT,p,weather,truth,True,final,next_plant)
            safety,_=full.short.proposal_gradient(full.ProposalPowerFlow(p),action,truth[1:])
            gradients.append(grad+WEIGHT*safety)
            total+=cost
        loss=(cmd*tensor(gradients)).sum(1).mean()
        indices=torch.as_tensor(draw.choice(exposures,128,replace=False),device=device)
        if arm!="frozen": loss=loss+10*(fore(history[indices])-targets[indices]).square().mean()
        opt.zero_grad()
        loss.backward()
        norms={}
        for name,network in (("actor",actor),("fore",fore)):
            norms[name]=float(torch.nn.utils.clip_grad_norm_(network.parameters(),3.,error_if_nonfinite=True))
        opt.step()
        visited+=len(batch);updates+=1
        if updates==1 or updates%50==0:
            record(folder/"progress.json",dict(stage="train",arm=arm,seed=seed,epoch=epoch,cycles=visited,
                   updates=updates,training_cycles=exposures,elapsed_s=perf_counter()-start,
                   observed_cost=total/visited,grad_norm=norms,device=torch.cuda.get_device_name(device)))
    assert visited==exposures==78694
    return updates


@torch.inference_mode()
def evaluate(actor,fore,split,folder,epoch,*,max_windows=None):
    # Evaluate all arms on the existing timed CPU deployment path.
    actor=copy.deepcopy(actor).cpu().eval()
    fore=copy.deepcopy(fore).cpu().eval()
    ref=base.direct.replay.forecast_net(base.direct.windows.REF)
    rows=[]
    for r,tr in data.evaluation_trajectories(split):
        last=tr.initial
        delays=[];kw=0.;v=o=vp=op=issued=0
        for i,h in enumerate(tr.d["past"]):
            p=tr.plant(i)
            cmd,status,delay,_=base.issue("direct",actor,fore,ref,p,last,h)
            result=data.hourly.scores(tr.model,p,last,cmd,delay,base.joint.phase_power(tr.d["truth"][i]),"direct")
            if cmd is not None: last=cmd
            delays.append(delay*1000);issued+=status=="issued"
            kw+=result["kwh"];v+=result["voltage_s"];o+=result["overload_s"]
            vp=max(vp,result["v_peak"]);op=max(op,result["overload_peak"])
        rows.append(dict(id=r.id,date=r.date,kwh=kw,voltage_s=v,overload_s=o,vpeak=vp,opeak=op,
                         issued=issued,ms_p50=np.median(delays),ms_p95=np.quantile(delays,.95)))
        if max_windows is not None and len(rows) >= max_windows:
            break
    f=pd.DataFrame(rows)
    f.to_csv(folder/f"{split}_{epoch}.csv",index=False)
    m=f.groupby("date").mean(numeric_only=True).mean()
    return dict(trade=float(m.kwh/1.5/100+WEIGHT*(.1*m.voltage_s/5400+50*m.vpeak+.2*m.overload_s/5400+20*m.opeak)),
                kwh=m.kwh,voltage_s=m.voltage_s,overload_s=m.overload_s)


def worker(arm,seed,epoch):
    folder=OUT/f"{arm}_{seed}"
    folder.mkdir(parents=True,exist_ok=True)
    result=folder/f"epoch_{epoch}.json"
    if result.exists(): return
    ds=data.ExpandedTrainingSet()
    actor,fore=base.networks(seed,ds.legacy.model)
    device=torch.device("cuda:0")
    actor.to(device);fore.to(device).requires_grad_(arm!="frozen")
    opt=torch.optim.Adam([{"params":actor.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    def snapshot(n):
        return folder/f"weights_{n}.pt"
    if epoch==1 and not snapshot(0).exists():
        full.atomic_save(dict(actor=actor.state_dict(),fore=fore.state_dict(),opt=opt.state_dict()),snapshot(0))
    if not snapshot(epoch).exists():
        s=torch.load(snapshot(epoch-1),map_location=device,weights_only=True)
        actor.load_state_dict(s["actor"]);fore.load_state_dict(s["fore"]);opt.load_state_dict(s["opt"])
        updates=fit_epoch(ds,actor,fore,opt,arm,seed,epoch,device,folder)
        full.atomic_save(dict(actor=actor.state_dict(),fore=fore.state_dict(),opt=opt.state_dict(),updates=updates),snapshot(epoch))
    for n in ([0,1] if epoch==1 else [epoch]):
        if (folder/f"epoch_{n}.json").exists(): continue
        s=torch.load(snapshot(n),map_location=device,weights_only=True)
        actor.load_state_dict(s["actor"]);fore.load_state_dict(s["fore"])
        val=evaluate(actor,fore,"validation",folder,n)
        record(folder/f"epoch_{n}.json",dict(epoch=n,updates=s.get("updates",0),**val))


def launch(gpus):
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/"driver.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=dict(dataset_sha=base.joint.sha(data.OUT/"lock.json"),arms=ARMS,seeds=SEEDS,weight=WEIGHT,
                  batch=BATCH,epochs="2 to 4 shared per seed. Stop after an epoch with no new validation best in any arm (1e-6).",
                  selector="Lowest date-balanced validation trade, including initialization; earliest tie.",
                  deployment="Identical timed CPU action issue and AC replay, training networks on CUDA.")
    if (OUT/"protocol.json").exists(): assert json.loads((OUT/"protocol.json").read_text())==json.loads(json.dumps(protocol))
    record(OUT/"protocol.json",protocol)
    def job(args):
        arm,seed,stage,epoch=args
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpus[SEEDS.index(seed)%len(gpus)]))
        with (OUT/f"{arm}_{seed}.log").open("a") as log:
            run([sys.executable,"-u",__file__,stage,arm,str(seed),str(epoch)],env=env,stdout=log,stderr=log,check=True)
    active=set(SEEDS); picked={}
    with ThreadPoolExecutor(max_workers=9) as pool:
        for epoch in range(1,5):
            list(pool.map(job,[(a,s,"worker",epoch) for s in sorted(active) for a in ARMS]))
            for seed in sorted(active):
                improved=False
                for arm in ARMS:
                    folder=OUT/f"{arm}_{seed}"
                    costs=[json.loads((folder/f"epoch_{i}.json").read_text())["trade"] for i in range(epoch+1)]
                    picked[arm,seed]=int(np.argmin(costs))
                    improved |= costs[-1]<min(costs[:-1])-1e-6
                if epoch>=2 and (not improved or epoch==4): active.remove(seed)
            if not active: break
        record(OUT/"selection.json",{f"{a}_{s}":n for (a,s),n in picked.items()})
        list(pool.map(job,[(a,s,"test",n) for (a,s),n in picked.items()]))
    record(OUT/"complete.json",{"paired_runs":len(picked),"test_splits":["test_may","test_september"]})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["launch","worker","test"])
    p.add_argument("params",nargs="*")
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=="launch": launch([int(i) for i in args.params])
    else:
        arm,seed,epoch=args.params
        seed,epoch=int(seed),int(epoch)
        if args.stage=="worker": worker(arm,seed,epoch)
        else:
            folder=OUT/f"{arm}_{seed}"
            actor,fore=base.networks(seed,base.plant_model())
            state=torch.load(folder/f"weights_{epoch}.pt",map_location="cpu",weights_only=True)
            actor.load_state_dict(state["actor"]);fore.load_state_dict(state["fore"])
            for split in ("test_may","test_september"):
                evaluate(actor,fore,split,folder,epoch)
