"""Three paired action-network arms for the Chen--Lin HPT comparison.

B freezes PV, M continues MSE, C couples downstream feedback to PV.
The forecast used to accept all suggestions is always the archived seed-11 MSE model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

from ieee123_joint5_trajectories import OUT as APRIL, joint, verify_file
from ieee123_preference_replay import features
from ieee123_ramp_dispatch import Snapshot
from ieee123_service_areas import metrics
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

OUT = APRIL.with_name("ieee123_chen_three_arm_v1")/"training"
STEPS, EVERY, PATIENCE = 120, 10, 40
STREAMS, FRAGMENT, DIRECTIONS = 2, 12, 4
MARGIN, EPSILON = .003, .002


def disk(x):
    a=np.clip(np.asarray(x,dtype=float).reshape(-1,18,3),-1,1).copy()
    uv=a[:,:,:2]
    uv/=np.maximum(1,np.linalg.norm(uv,axis=2,keepdims=True))
    return a.reshape(-1,54)


class DispatchActor(nn.Module):
    def __init__(self,seed,base=None):
        super().__init__()
        self.base=joint.Action(seed)
        if base is not None:
            self.base.load_state_dict(base)
        self.equipment=nn.Sequential(nn.Linear(15,16),nn.SiLU(),nn.Linear(16,9))
        nn.init.zeros_(self.equipment[-1].weight)
        nn.init.zeros_(self.equipment[-1].bias)

    def forward(self,x,past,path,local):
        base=self.base(x,path,past[:,-1],x.new_full((len(x),),.5))
        a=(base.reshape(-1,6,9)+.25*self.equipment(local).tanh()).reshape(-1,18,3).clamp(-1,1)
        uv=a[:,:,:2]
        return torch.cat((uv/uv.norm(dim=2,keepdim=True).clamp_min(1),a[:,:,2:]),2).flatten(1)


def local_features(plant,old,past):
    m=plant.m
    pv=joint.phase_power(past[-1])
    state=m.solve(plant.slot,old,"high",plant.taps,generation_kw=pv,tolerance=1e-8,max_iterations=1500)
    v=state["v"][m.secondary]
    pu=np.abs(v)/m.bases[m.secondary]
    pu[0]=np.abs(v[0]-np.roll(v[0],-1))/480
    # Per-phase port loading uses the actual series/shunt converter powers.
    spare=1-state["ports"].max(axis=0)
    return np.column_stack((old.reshape(6,9),(pu-1)*20,spare))


def load_policy(arm,seed):
    actor=DispatchActor(seed)
    w=torch.load(OUT/f"{arm}_{seed}.pt",weights_only=True)
    actor.load_state_dict(w["actor"])
    actor.eval()
    @torch.inference_mode()
    def act(plant,old,past,path):
        feat=features(plant.m,plant.slot,plant.taps,past)
        values=[joint.tensor(np.asarray(v)[None]) for v in (feat,past,path,local_features(plant,old,past))]
        return disk(actor(*values).numpy())[0]
    return act


def eval_rows(plant,actions,powers):
    try:
        return plant.evaluate(actions,powers)
    except ArithmeticError:
        return np.concatenate([plant.evaluate(a,g,max_iterations=1500) for a,g in zip(actions,powers)])


def scores(plant,commands,weather):
    """Batched forecast screen plus cold check, using the deployment numerical tolerances."""
    n=2*plant.m.original_n
    powers=np.tile(weather,(len(commands),1)); actions=np.repeat(commands,len(weather),axis=0)
    y=eval_rows(plant,actions,powers).reshape(len(commands),len(weather),-1)
    out=[]
    for x,frames in zip(commands,y):
        c=frames[:,1:].min(0)
        physical=c.min()>=0
        def near(raw):
            return (raw[:n].min()>=0 and raw[-54:].min()>=0 and
                    1-(1+1e-7)**2<=raw[n:-54].min()<0)
        if physical:
            raw=Snapshot(plant.m,plant.slot,plant.taps,weather[0]).evaluate(x)[0][1:]
            physical=raw.min()>=0
            boundary=not physical and near(raw)
        else:
            boundary=near(c)
        if boundary:
            physical=True
            for g in np.unique(weather,axis=0):
                try:
                    st=plant.m.solve(plant.slot,x,"high",plant.taps,generation_kw=g,tolerance=1e-10,max_iterations=1500)
                except ArithmeticError:
                    physical=False
                    break
                m=metrics(plant.m,st,plant.slot,"high","train",plant.m.ratings.ravel())
                if not (.95<=m["vmin"] and m["vmax"]<=1.05 and m["main_util"]<=1+1e-7 and m["port_util"]<=1):
                    physical=False
                    break
        out.append((max(0.,MARGIN-c[:n].min()),frames[:,0].mean()) if physical else None)
    return out


def better(candidate,reference):
    return candidate is not None and (reference is None or candidate[0]<reference[0]-1e-9 or
                 (abs(candidate[0]-reference[0])<=1e-9 and candidate[1]<reference[1]-1e-7))


def outcomes(plant,old,fixed,proposals,weather,truth):
    bank=np.vstack((old,fixed,proposals))
    ranks=scores(plant,bank,weather)
    backup=1 if better(ranks[1],ranks[0]) else 0
    winners=np.array([j if better(ranks[j],ranks[backup]) else backup for j in range(2,len(bank))])
    # At the first measured second the previous command is still active. For training,
    # all timely proposals use the same one-sample actuation lag. Timed deployment is separate.
    unique=np.unique(np.r_[0,winners])
    y=eval_rows(plant,np.repeat(bank[unique],5,axis=0),np.tile(truth,(len(unique),1))).reshape(len(unique),5,-1)
    actual=y[np.searchsorted(unique,winners)].copy()
    actual[:,0]=y[0,0]
    n=2*plant.m.original_n
    gap=(-actual[:,:,1:1+n].min(2)).clip(0)
    util=(1-actual[:,:,1+n:-18].min(2)).clip(0)**.5
    cost=actual[:,:,0].mean(1)
    risk=np.column_stack(((gap>0).mean(1),gap.max(1),(util>1+1e-7).mean(1),(util-1).clip(0).max(1)))
    return bank[winners],cost,risk,winners>=2


class April:
    def __init__(self):
        verify_file("cases.npz")
        self.d=dict(np.load(APRIL/"cases.npz"))
        valid=np.isfinite(self.d["target"]).all(axis=(1,2))
        self.val=np.flatnonzero(valid&(self.d["date"]>20100422))
        self.train=np.flatnonzero(valid&(self.d["date"]<=20100422))
        self.starts=np.arange(len(self.d["slot"])-len(self.d["applied"]),len(self.d["slot"]),120)
        assert len(self.val)==36 and len(self.train)==5254
        assert all(np.all(np.diff(self.d["second"][s:s+120])==5) and
                   np.unique(self.d["date"][s:s+120]).size==1 and self.d["date"][s]<=20100422 for s in self.starts)
        self.fixed=np.load(joint.SOURCE/"windows.npz")["fixed"]
        _,self.model,*_=joint.build()
        ref,_=joint.load_models(11,joint.OUT/"init_11.pt")
        with torch.inference_mode():
            self.ref=ref(joint.tensor(self.d["past"])).numpy()
        self.plants={}

    def plant(self,i):
        key=(int(self.d["slot"][i]),tuple(self.d["taps"][i]))
        if key not in self.plants:
            self.plants[key]=joint.BatchAC(self.model,*key)
        return self.plants[key]

    def case(self,i,old):
        past=self.d["past"][i]
        return (self.plant(i),joint.tensor(self.d["feat"][i:i+1]),joint.tensor(past[None]),
             joint.tensor(local_features(self.plant(i),old,past)[None]),
             joint.phase_power(np.vstack((past[-1],self.ref[i]))),joint.phase_power(self.d["truth"][i]))


def op_values(cost,risk):
    # Dimensionless energy at a 100-kW base plus frequency and peak-severity terms.
    return cost/100 + risk@np.array([.1,50.,.2,20.])


def train(arm,seed,data):
    path=OUT/f"{arm}_{seed}"
    if path.with_suffix(".json").exists():
        meta=json.loads(path.with_suffix(".json").read_text())
        assert joint.sha(path.with_suffix(".pt"))==meta["checkpoint_sha"]
        return
    t0=perf_counter()
    torch.manual_seed(seed)
    initial=joint.OUT/f"init_{seed}.pt"
    fore,base=joint.load_models(seed,initial)
    actor=DispatchActor(seed,base.state_dict())
    initial_actor=copy.deepcopy(actor.state_dict());initial_fore=copy.deepcopy(fore.state_dict())
    fore.requires_grad_(arm!="B")
    opt=torch.optim.Adam([{"params":actor.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    rng=np.random.default_rng(seed)
    h=joint.tensor(data.d["past"])
    target=joint.tensor(data.d["target"])
    updates=[];history=[];selections=[];seen=set();best=None
    for step in range(STEPS+1):
        if step%EVERY==0:
            with torch.inference_mode():
                val=[]
                for i in data.val:
                    old=data.d["old"][i];p,x,past,loc,weather,truth=data.case(i,old)
                    raw=disk(actor(x,past,fore(past),loc).numpy())
                    _,kw,risk,won=outcomes(p,old,data.fixed,raw,weather,truth)
                    val.append([kw[0],*risk[0],won[0]])
                kw,v,gap,o,peak,won=np.mean(val,axis=0)
            # Risk-cap criterion relative to the matched initialization. Tied at cap:
            # minimum energy. Fixed limits and same validation rows for B, M, C.
            if step==0:
                caps=np.array([v,gap,o,peak])
            exceeded=(np.array([v,gap,o,peak])-caps-1e-8).clip(0)
            key=(float(exceeded@[1,50,2,20]),float(kw))
            if best is None or key<best:
                best=key;pick=step
                torch.save({"fore":fore.state_dict(),"actor":actor.state_dict()},path.with_suffix(".pt"))
            history.append(dict(step=step,kw=kw,voltage_rate=v,mean_max_gap=gap,overload_rate=o,
                        mean_peak_excess=peak,proposal_fraction=won,risk_excess=key[0]))
            if step==STEPS or step>=40 and step-pick>=PATIENCE:
                break
        if step%FRAGMENT==0:
            idx=rng.choice(data.starts,STREAMS,replace=False)+rng.integers(0,121-FRAGMENT,STREAMS)
            carried=[data.d["old"][i].copy() for i in idx]
        terms=[];wincount=0;changes=0
        for j,i0 in enumerate(idx):
            i=int(i0+step%FRAGMENT)
            old=carried[j];p,x,past,loc,weather,truth=data.case(i,old)
            f=fore(past)
            cmd=actor(x,past,f if arm=="C" else f.detach(),loc)
            raw=disk(cmd.detach().numpy())[0]
            dirs=rng.choice([-1.,1.],(DIRECTIONS,54))
            probes=disk(np.vstack((raw,raw+EPSILON*dirs,raw-EPSILON*dirs)))
            choices,cost,risk,wins=outcomes(p,old,data.fixed,probes,weather,truth)
            vals=op_values(cost,risk)
            slope=(vals[1:1+DIRECTIONS]-vals[1+DIRECTIONS:])/(2*EPSILON)
            grad=joint.tensor(slope@dirs/DIRECTIONS)
            terms.append(joint.tensor(vals[0])+(grad*(cmd[0]-cmd[0].detach())).sum())
            carried[j]=choices[0].copy()
            updates.append(dict(step=step+1,case=i,stream=j,proposal_won=int(wins[0]),kw=cost[0],
                                voltage_rate=risk[0,0],overload_rate=risk[0,2],grad_norm=float(grad.norm())))
            seen.add(i); changes+=int(np.count_nonzero(wins[1:]!=wins[0]));wincount+=int(wins[0])
        ids=rng.choice(data.train,128,replace=False)
        mse=(fore(h[ids])-target[ids]).square().mean()
        opt.zero_grad()
        torch.stack(terms).mean().backward()
        gnorm=sum(p.grad.square().sum().item() for p in fore.parameters() if p.grad is not None)**.5
        if arm!="B":
            (10*mse).backward()
        for net in (fore,actor):
            nn.utils.clip_grad_norm_(net.parameters(),3.,error_if_nonfinite=True)
        opt.step()
        selections.append(dict(step=step+1,fore_task_grad=gnorm,mse=float(mse.detach()),won=wincount,
                               perturbation_gate_flips=changes))
    pd.DataFrame(history).to_csv(path.parent/f"{path.name}_validation.csv",index=False)
    pd.DataFrame(updates).to_csv(path.parent/f"{path.name}_cases.csv",index=False)
    pd.DataFrame(selections).to_csv(path.parent/f"{path.name}_gradients.csv",index=False)
    saved=torch.load(path.with_suffix(".pt"),weights_only=True)
    diff=lambda a,b: max((a[k]-b[k]).abs().max().item() for k in a)
    peak_task_grad=max(r["fore_task_grad"] for r in selections)
    assert (peak_task_grad==0 if arm!="C" else peak_task_grad>0)
    if arm=="B":
        assert diff(saved["fore"],initial_fore)==0
    meta=dict(arm=arm,seed=seed,selected=pick,stopped=step,available_train=len(data.train),val=len(data.val),
              used_train=len(seen),wall_s=perf_counter()-t0,fore_task_gradient_max=peak_task_grad,
              selected_fore_change=diff(saved["fore"],initial_fore),selected_actor_change=diff(saved["actor"],initial_actor),
              terminal_fore_change=diff(fore.state_dict(),initial_fore),terminal_actor_change=diff(actor.state_dict(),initial_actor),
              reference=joint.sha(joint.OUT/"init_11.pt"),init_sha=joint.sha(initial),
              checkpoint_sha=joint.sha(path.with_suffix(".pt")),code_sha=joint.sha(Path(__file__)),
              data_sha=joint.sha(APRIL/"cases.npz"),val_caps=caps.tolist(),
              sampled_ids_sha=hashlib.sha256(np.asarray([r["case"] for r in updates],dtype="<i8").tobytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    print(arm,seed,"selected",pick,"stop",step,"seconds",round(meta["wall_s"],1),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--arms",nargs="+",choices=list("BCM"),default=list("BCM"))
    ap.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        data=April()
        for seed in args.seeds:
            for arm in args.arms:
                train(arm,seed,data)
