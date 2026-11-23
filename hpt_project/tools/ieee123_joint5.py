"""Causal five-second PV forecasts, action-side and joint DFL on the six-HPT feeder."""
from ieee123_preference_replay import OUT as PREVIOUS, SOURCE, SEEDS, MARGINS, fixed_data, run, sha
from ieee123_interval_ac import BatchAC
from ieee123_interval_data import MAY, phase_power
from ieee123_ramp_windows import ARCHIVE, SIX, day_data
from ieee123_service_areas import build
from ieee123_preference_train import CAPS, TAIL
from pathlib import Path
import copy
import json
import sys
import zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

OUT=SOURCE.with_name("ieee123_joint_five_second_v1")
ARMS=("frozen5","mse5","joint5")
RISK=torch.arange(3,dtype=torch.float32)/2

def tensor(a):
    return torch.as_tensor(a,dtype=torch.float32)

def prepare():
    OUT.mkdir(exist_ok=True)
    data=dict(np.load(PREVIOUS/"april_examples.npz"))
    assert sha(PREVIOUS/"april_examples.npz")==json.loads((PREVIOUS/"preference_11.json").read_text())["april_sha256"]
    assert sha(ARCHIVE)==json.loads((SOURCE/"lock.json").read_text())["april_sha"]
    histories=[];targets=[];dates=[]
    truth=np.full((len(data["slot"]),5,6),np.nan)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            date=int(name[:8]);p=day_data(archive,name)[:,SIX]
            for t in range(36000,50400,30):
                block=p[t-10:t+6]
                if np.isfinite(block).all():
                    histories.append(block[:11]);targets.append(block[11:]);dates.append(date)
            for i in np.flatnonzero(data["date"]==date):
                t=data["second"][i]
                np.testing.assert_array_equal(data["past"][i],p[t-10:t+1])
                np.testing.assert_array_equal(data["truth"][i],p[t:t+5])
                truth[i]=p[t+1:t+6]
    np.savez_compressed(OUT/"april.npz",h=histories,y=targets,date=dates,physical_truth=truth)

class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(66,64),nn.SiLU(),nn.Linear(64,30))

    def forward(self,h):
        latest=h[:,-1:]
        x=torch.cat((latest,5*(h[:,1:]-h[:,:-1])),1)
        return (latest+.35*self.net(x.flatten(1)).tanh().reshape(-1,5,6)).clamp(0,1)

class Action(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with np.load(PREVIOUS/f"preference_{seed}.npz") as z:
            assert sha(PREVIOUS/f"preference_{seed}.npz")==json.loads((PREVIOUS/f"preference_{seed}.json").read_text())["weights_sha256"]
            self.register_buffer("mean",tensor(z["mean"]))
            self.register_buffer("scale",tensor(z["scale"]))
            self.body=nn.Sequential(nn.Linear(len(self.mean)+1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.action=nn.Linear(128,54)
            self.load_state_dict({k:tensor(z[k]) for k in z.files})
        self.future=nn.Linear(30,64,bias=False)
        nn.init.normal_(self.future.weight,std=.025)

    def forward(self,x,trajectory,latest,risk):
        h=torch.cat(((x-self.mean)/self.scale,risk[:,None]),1)
        # The 30 ordered site/horizon changes enter before the first activation.
        h=self.body[0](h)+self.future(5*(trajectory-latest[:,None,:]).flatten(1))
        h=self.body[3](self.body[2](self.body[1](h)))
        a=self.action(torch.cat((h,h*risk[:,None]),1)).tanh().reshape(-1,18,3)
        uv=a[:,:,:2]
        return torch.cat((uv/uv.norm(dim=2,keepdim=True).clamp_min(1),a[:,:,2:]),2).flatten(1)

def pretrain(z,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    h,y=tensor(z["h"]),tensor(z["y"])
    train=np.flatnonzero(z["date"]<=20100422);val=np.flatnonzero(z["date"]>20100422)
    f=Forecaster();opt=torch.optim.Adam(f.parameters(),lr=.002)
    hist=[];best=np.inf
    for epoch in range(80):
        for ids in np.array_split(rng.permutation(train),int(np.ceil(len(train)/256))):
            loss=(f(h[ids])-y[ids]).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():v=float((f(h[val])-y[val]).square().mean())
        hist.append(v)
        if v<best:
            best=v;pick=epoch;weights=copy.deepcopy(f.state_dict())
        if epoch-pick>=8:break
    f.load_state_dict(weights);a=Action(seed)
    torch.save({"fore":weights,"action":a.state_dict()},OUT/f"init_{seed}.pt")
    pd.DataFrame({"val_mse":hist}).to_csv(OUT/f"pretrain_{seed}.csv",index=False)

def load_models(seed,path):
    f=Forecaster();a=Action(seed)
    w=torch.load(path,weights_only=True)
    f.load_state_dict(w["fore"]);a.load_state_dict(w["action"])
    return f,a

class TrainingCases:
    def __init__(self,m,d,truth,valid):
        self.nv=2*m.original_n
        self.plants={};self.backup={};self.keys={}
        self.truth=phase_power(truth);self.current=phase_power(d["past"][:,-1])
        fixed=np.load(SOURCE/"windows.npz")["fixed"]
        for i in valid:
            p=BatchAC(m,d["slot"][i],tuple(d["taps"][i]))
            self.plants[i]=p
            options=np.stack((d["old"][i],fixed))
            y=p.evaluate(options,self.current[i])
            for mode in range(3):
                feasible=y[:,1:].min(1)>=0
                deficit=np.maximum(0,MARGINS[mode]-y[:,1:1+self.nv].min(1))
                choice=min(np.flatnonzero(feasible),key=lambda k:(deficit[k],y[k,0])) if feasible.any() else 0
                self.backup[i,mode]=options[choice]
                self.keys[i,mode]=(deficit[choice],y[choice,0]) if feasible.any() else (np.inf,np.inf)

    def winners(self,i,actions):
        y=self.plants[i].evaluate(actions,self.current[i])
        deficit=np.maximum(0,np.asarray(MARGINS)-y[:,1:1+self.nv].min(1))
        best,cost=np.array([self.keys[i,k] for k in range(3)]).T
        return (y[:,1:].min(1)>=0)&((deficit<best-1e-9)|((abs(deficit-best)<=1e-9)&(y[:,0]<cost-1e-7)))

    def measures(self,y):
        return np.stack((y[:,0],-y[:,1:1+self.nv].min(1),-y[:,1+self.nv:-18].min(1)),axis=1)

    def execute(self,i,a):
        won=self.winners(i,a)
        chosen=np.where(won[:,None],a,np.stack([self.backup[i,k] for k in range(3)]))
        return chosen,won

def train(seed,arm,z,d,cases,tr,va,*,output=OUT,step_limit=120,fixed_steps=False):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    init=OUT/f"init_{seed}.pt";f,a=load_models(seed,init)
    h,x=tensor(d["past"]),tensor(d["feat"])
    fh,fy=tensor(z["h"]),tensor(z["y"])
    ids_fore=np.flatnonzero(z["date"]<=20100422)
    f.requires_grad_(arm!="frozen5")
    opt=torch.optim.AdamW([{"params":a.parameters(),"lr":.0005},{"params":f.parameters(),"lr":.00015}],weight_decay=.0001)
    dual=np.array([.4,1.,2.]);tail=np.array([.05,.25,.75])
    history=[];updates=[];best=None;draws=[]
    def predict(ids):
        forecast=f(h[ids])
        use=forecast if arm=="joint5" else forecast.detach()
        return a(x[ids].repeat_interleave(3,0),use.repeat_interleave(3,0),
                 h[ids,-1].repeat_interleave(3,0),RISK.repeat(len(ids)))
    for step in range(step_limit+1):
        if step%10==0:
            with torch.no_grad():actions=predict(va).numpy().reshape(-1,3,54).astype(float)
            values=[]
            for i,cmd in zip(va,actions):
                chosen,_=cases.execute(i,cmd)
                y=cases.plants[i].evaluate(np.repeat(chosen,5,axis=0),np.tile(cases.truth[i],(3,1)))
                values.append(cases.measures(y).reshape(3,5,3))
            val=np.stack(values);v=(val[:,:,:,1]>0).mean((0,2))
            key=(float((val[:,:,:,2]>2e-7).mean()),float(np.maximum(0,v-CAPS).sum()),float(val[:,:,:,0].mean()))
            history.append([step,*key])
            if best is None or key<best:
                best=key;pick=step
                torch.save({"fore":f.state_dict(),"action":a.state_dict()},output/f"{arm}_{seed}.pt")
            if (not fixed_steps and step-pick>=40) or step==step_limit:break
        ids=rng.choice(tr,8,replace=False);hz=rng.integers(0,5,len(ids))
        draws.extend(ids.tolist())
        batch=rng.choice(ids_fore,128,replace=False)
        cmd=predict(ids);raw=cmd.detach().numpy().reshape(-1,3,54).astype(float)
        vals=[];jac=[];nwon=0
        for j,i in enumerate(ids):
            chosen,won=cases.execute(i,raw[j]);nwon+=won.sum()
            dirs=rng.choice([-1.,1.],(8,54));eps=.0002
            trials=np.concatenate((chosen[:,None],chosen[:,None]+eps*dirs,chosen[:,None]-eps*dirs),axis=1)
            y=cases.measures(cases.plants[i].evaluate(trials.reshape(-1,54),cases.truth[i,hz[j]])).reshape(3,17,3)
            vals.extend(y[:,0])
            # Piecewise derivative of issued action at the current winner. Backups have zero derivative.
            jac.extend(np.einsum("rkc,kd->rcd",y[:,1:9]-y[:,9:],dirs)/(16*eps)*won[:,None,None])
        feedback=tensor(np.asarray(vals))+torch.einsum("ncd,nd->nc",tensor(np.asarray(jac)),cmd-cmd.detach())
        kw,gap,equip=feedback.reshape(-1,3,3).unbind(2)
        hard=(gap>0).float();soft=(gap/.002).sigmoid()
        worst=gap.relu().max(0).values
        op=(kw.mean(0)/100+tensor(dual)*(hard+soft-soft.detach()).mean(0)+tensor(tail)*worst/.01).mean()+40*equip.relu().mean()
        dual=np.clip(dual+.04*(hard.mean(0).numpy()-CAPS),0,12)
        tail=np.clip(tail+.025*(worst.detach().numpy()-TAIL)/.01,0,12)
        mse=(f(fh[batch])-fy[batch]).square().mean()
        opt.zero_grad();op.backward()
        gnorm=float(sum(p.grad.square().sum().item() for p in f.parameters() if p.grad is not None)**.5)
        if arm!="frozen5":(10*mse).backward()
        # Clip modules independently so enabling forecast gradients cannot shrink the actor update.
        for module in (a,f):nn.utils.clip_grad_norm_(module.parameters(),3.,error_if_nonfinite=True)
        opt.step();updates.append([step+1,gnorm,nwon,float(mse.detach())])
    stem=output/f"{arm}_{seed}"
    pd.DataFrame(history,columns=["step","overload_rate","excess_frequency","kw"]).to_csv(stem.with_suffix(".csv"),index=False)
    pd.DataFrame(updates,columns=["step","fore_dispatch_grad","won_of_24","anchor_mse"]).to_csv(output/f"{stem.name}_updates.csv",index=False)
    norms=np.array(updates)[:,1]
    if arm!="joint5":assert norms.max()==0
    saved=torch.load(stem.with_suffix(".pt"),weights_only=True)
    initial=torch.load(init,weights_only=True)
    diff=max(float((v-initial["fore"][k]).abs().max()) for k,v in saved["fore"].items())
    if arm=="frozen5":assert diff==0
    stem.with_suffix(".json").write_text(json.dumps(dict(selected=pick,stopped=step,initial_sha=sha(init),
        selected_sha=sha(stem.with_suffix(".pt")),fore_max_change=diff,peak_fore_dispatch_grad=norms.max(),
        draws=len(draws),unique_states=len(set(draws)),train=len(tr),val=len(va),forecast_train=len(ids_fore),forecast_val=len(fh)-len(ids_fore))))
    print("selected",arm,seed,pick,step,flush=True)

def loader(arm,seed,output=OUT):
    stem=output/f"{arm}_{seed}"
    assert sha(stem.with_suffix(".pt"))==json.loads(stem.with_suffix(".json").read_text())["selected_sha"]
    f,a=load_models(seed,stem.with_suffix(".pt"))
    @torch.no_grad()
    def act(past,x,mode):
        h=tensor(past[None])
        return a(tensor(x[None]),f(h),h[:,-1],RISK[mode:mode+1]).numpy()[0].astype(float)
    act.checkpoint_sha=sha(stem.with_suffix(".pt"))
    return act

def main(stage):
    torch.set_num_threads(1)
    if stage=="train":
        prepare()
        z=dict(np.load(OUT/"april.npz"));d=dict(np.load(PREVIOUS/"april_examples.npz"))
        valid=np.isfinite(z["physical_truth"]).all(axis=(1,2))
        tr=np.flatnonzero(valid&(d["date"]<=20100422));va=np.flatnonzero(valid&(d["date"]>20100422))
        _,m,*_=build();cases=TrainingCases(m,d,z["physical_truth"],np.flatnonzero(valid))
        for seed in SEEDS:
            pretrain(z,seed)
            for arm in ARMS:train(seed,arm,z,d,cases,tr,va)
    elif stage=="replay":run(ARMS,policy_loader=loader,output=OUT)
    else:
        from ieee123_joint5_report import report
        report()

if __name__=="__main__":
    with threadpool_limits(1):main(sys.argv[1])
