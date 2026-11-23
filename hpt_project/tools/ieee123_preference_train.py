"""April-only teachers, physical gradients and calibration for the 54-output head."""
from ieee123_preference_replay import OUT,SOURCE,SEEDS,LEARNED,MARGINS,features,sha,policy
from ieee123_interval_run import dispatch, BatchAC, build, phase_power
from ieee123_ramp_windows import ARCHIVE,SIX,day_data,ROOT
from ieee123_service_areas import local,ZERO
from datetime import datetime
import argparse
import json
import sys
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

CAPS=np.array([.25,.10,.025]) # fraction of seconds allowed in April checkpoint selection
TAIL=np.array([.012,.006,.002])
MAX_STEPS=240
PATIENCE=60
EVAL_EVERY=15

def dataset(m):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"april_examples.npz"
    if path.exists():return dict(np.load(path))
    sys.path.insert(0,str(ROOT/"src"))
    from ieee123_fast_ramps import predictor
    archived=json.loads((SOURCE/"lock.json").read_text())
    assert sha(ARCHIVE)==archived["april_sha"]
    rows=[]
    rng=np.random.default_rng(123)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            p=day_data(archive,name)
            date=name[:8]
            for block in np.array_split(np.arange(36000,50400,900),6):
                t=int(rng.choice(block))+120
                before=p[(t//900)*900-3600:(t//900)*900].reshape(4,900,9)
                short=p[t-10:t+5,SIX]
                if not np.isfinite(short).all() or not np.isfinite(before).all():continue
                slot=(datetime.strptime(date,"%Y%m%d").timetuple().tm_yday-1)*96+t//900
                clock=2*np.pi*slot/96
                hist=np.r_[before.mean(1)[::-1].ravel(),np.sin(clock),np.cos(clock)]
                rows.append((int(date),slot,hist,short,t))
    slow,hashes=predictor(np.stack([r[2] for r in rows]))
    assert hashes==archived["predictors"]
    arrays={k:[] for k in ("date","slot","second","taps","old","past","truth","feat")}
    for (date,slot,_,short,t),pv in zip(rows,slow):
        st=m.step(slot,ZERO,"high",generation_kw=phase_power(pv))
        taps=st["taps"]
        old=local(st,m,ZERO)
        for k,value in zip(arrays,(date,slot,t,taps,old,short[:11],short[10:],features(m,slot,taps,short[:11]))):
            arrays[k].append(value)
    np.savez_compressed(path,**arrays)
    return dict(np.load(path))

def teachers(m,d):
    fixed=np.load(SOURCE/"windows.npz")["fixed"]
    folder=OUT/"teachers";folder.mkdir(exist_ok=True)
    for i in range(len(d["slot"])):
        path=folder/f"{i:03d}.npz"
        if path.exists():continue
        plant=BatchAC(m,d["slot"][i],tuple(d["taps"][i]))
        acts=[]; ok=[]
        for margin in MARGINS:
            events,_=dispatch("snapshot",plant,d["old"][i],fixed,d["past"][i],None,None,margin=margin)
            acts.append(events[-1][1] if events else d["old"][i]);ok.append(bool(events))
        np.savez_compressed(path,a=acts,ok=ok)
        if i%20==0:print("April teacher",i,len(d["slot"]),flush=True)

def train(m,d,methods):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    xt=torch.tensor(d["feat"],dtype=torch.float32)
    tr=np.flatnonzero(d["date"]<=20100422);va=np.flatnonzero(d["date"]>20100422)
    assert len(tr)>50 and len(va)>15 and set(d["date"][tr]).isdisjoint(d["date"][va])
    eligible=None
    if any(method.startswith("imitation") for method in methods):
        labels=[dict(np.load(OUT/"teachers"/f"{i:03d}.npz")) for i in range(len(xt))]
        teacher=torch.tensor(np.stack([r["a"] for r in labels]),dtype=torch.float32)
        eligible=np.stack([r["ok"] for r in labels])
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean",xt[tr].mean(0))
            self.register_buffer("scale",xt[tr].std(0).clamp_min(.05))
            self.body=nn.Sequential(nn.Linear(xt.shape[1]+1,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
            self.action=nn.Linear(128,54)
            nn.init.normal_(self.action.weight,std=.003);nn.init.zeros_(self.action.bias)
        def forward(self,x,r):
            h=self.body(torch.cat(((x-self.mean)/self.scale,r[:,None]),1))
            a=self.action(torch.cat((h,h*r[:,None]),1)).tanh().reshape(-1,18,3)
            uv=a[:,:,:2]
            return torch.cat((uv/uv.norm(dim=2,keepdim=True).clamp_min(1),a[:,:,2:]),2).reshape(-1,54)
    def plant(i):return BatchAC(m,d["slot"][i],tuple(d["taps"][i]))
    nv=2*m.original_n
    def measured(results):
        v=results[:,1:1+nv]
        equipment=results[:,1+nv:-18]
        return results[:,0],-v.min(1),-equipment.min(1)
    def validate(net):
        stats=[[] for _ in range(3)]
        with torch.no_grad():
            for i in va:
                a=net(xt[i:i+1].repeat(3,1),torch.arange(3)/2).numpy().astype(float)
                y=plant(i).evaluate(np.repeat(a,5,axis=0),np.tile(phase_power(d["truth"][i]),(3,1)))
                vals=np.stack(measured(y),1).reshape(3,5,3)
                for j in range(3):stats[j].extend(vals[j])
        s=np.array(stats)
        frequency=(s[:,:,1]>0).mean(1)
        over=(s[:,:,2]>2e-7).mean(1)
        energy=s[:,:,0].mean()
        return (float(over.sum()),float(np.maximum(0,frequency-CAPS).sum()),float(energy))
    for seed in SEEDS:
        for method in methods:
            base=OUT/f"{method}_{seed}"
            if base.with_suffix(".json").exists():continue
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            net=Net();opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
            dual=np.array([.4,1.,2.]);tail=np.array([.05,.25,.75])
            best=None;history=[]
            for step in range(MAX_STEPS+1):
                if step%EVAL_EVERY==0:
                    operating_key=validate(net)
                    key=operating_key
                    val_mse=np.nan
                    if method.startswith("imitation"):
                        with torch.no_grad():
                            pred=net(xt[va].repeat_interleave(3,0),torch.arange(3).repeat(len(va))/2)
                            mask=eligible[va].ravel()
                            val_mse=float((pred[mask]-teacher[va].reshape(-1,54)[mask]).square().mean())
                        if method=="imitation_mse": key=(val_mse,)
                    history.append([step,*operating_key,val_mse])
                    if best is None or key<best:
                        best=key;selected=step
                        w={k:v.detach().numpy().copy() for k,v in net.state_dict().items()}
                    if step-selected>=PATIENCE or step==MAX_STEPS:break
                ids=rng.choice(tr,8,replace=False)
                # Each weather sample sees all three priorities; common random numbers across arms.
                future=rng.integers(0,5,len(ids))
                r=torch.arange(3).repeat(len(ids))/2
                a=net(xt[ids].repeat_interleave(3,0),r)
                if method.startswith("imitation"):
                    mask=eligible[ids].ravel()
                    if not mask.any():continue
                    loss=(a[mask]-teacher[ids].reshape(-1,54)[mask]).square().mean()
                else:
                    vals=[];grads=[]
                    for n,i in enumerate(ids):
                        p=plant(i);g=phase_power(d["truth"][i,future[n]])
                        ac=a[3*n:3*n+3].detach().numpy().astype(float)
                        # Differentiate loss, worst voltage margin and worst equipment margin
                        # on actual April weather, through the exact six-HPT AC equations.
                        # Eight central Rademacher probes estimate the full VJP in expectation.
                        dirs=rng.choice([-1.,1.],(8,54))
                        queries=np.concatenate([ac[:,None,:],ac[:,None,:]+.0002*dirs,ac[:,None,:]-.0002*dirs],1)
                        y=np.stack(measured(p.evaluate(queries.reshape(-1,54),g)),1).reshape(3,17,3)
                        ghat=np.einsum("rkc,kd->rcd",y[:,1:9]-y[:,9:],dirs)/(.0004*len(dirs))
                        vals.extend(y[:,0]);grads.extend(ghat)
                    feedback=torch.tensor(np.array(vals),dtype=a.dtype)+torch.einsum("ncd,nd->nc",
                                torch.tensor(np.array(grads),dtype=a.dtype),a-a.detach())
                    kw,gap,over=feedback.reshape(-1,3,3).unbind(2)
                    soft=(gap/.002).sigmoid()
                    hard=(gap>0).float()
                    rate=(hard+soft-soft.detach()).mean(0)
                    tailgap=gap.relu().topk(1,dim=0).values.mean(0)
                    dp=torch.tensor(dual,dtype=a.dtype)
                    loss=(kw.mean(0)/100+dp*rate).mean()+40*over.relu().mean()
                    if method=="preference":
                        loss=loss+(torch.tensor(tail,dtype=a.dtype)*tailgap/.01).mean()
                        tail=np.clip(tail+.025*(tailgap.detach().numpy()-TAIL)/.01,0,12)
                    else:loss=loss+(15*dp*gap.relu().mean(0)).mean()
                    dual=np.clip(dual+.04*(hard.mean(0).numpy()-CAPS),0,12)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),3,error_if_nonfinite=True);opt.step()
            np.savez(base.with_suffix(".npz"),**w)
            net.load_state_dict({k:torch.tensor(v) for k,v in w.items()})
            with torch.no_grad():
                np.testing.assert_allclose(policy(w,d["feat"][va[0]],1),net(xt[va[:1]],torch.tensor([.5])).numpy()[0],atol=2e-6)
            pd.DataFrame(history,columns=["step","sum_overload_rate","sum_rate_excess","mean_kw","teacher_mse"]).to_csv(base.with_suffix(".csv"),index=False)
            base.with_suffix(".json").write_text(json.dumps(dict(seed=seed,selected=selected,stopped=step,
                weights_sha256=sha(base.with_suffix(".npz")),april_sha256=sha(OUT/"april_examples.npz"),train=len(tr),validation=len(va),
                eligible_teacher_labels=int(eligible[tr].sum()) if eligible is not None else None,
                caps=CAPS.tolist(),tail_targets=TAIL.tolist())))
            print("locked",method,seed,selected,step,flush=True)

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("stage",choices=["prepare","train"])
    args.add_argument("methods",nargs="*")
    opt=args.parse_args()
    with threadpool_limits(1):
        _,model,*_=build()
        data=dataset(model)
        if opt.stage=="prepare":teachers(model,data)
        else:train(model,data,opt.methods or LEARNED)
