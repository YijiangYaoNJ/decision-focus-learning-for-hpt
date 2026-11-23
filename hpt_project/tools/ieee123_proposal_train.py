"""Add a realized proposal-safety derivative to the two-network IEEE123 controller."""
import argparse
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import ieee123_direct_confirm as base
from ieee123_proposal_windows import OUT, WEIGHTS

prior=base.prior
tensor=base.joint.tensor
RISK=np.array([.1,50.,.2,20.])


def components(plant, y):
    n=2*plant.m.original_n
    voltage=-y[:,1:1+n].min(1)
    capacity=np.sqrt(np.maximum(0.,1-y[:,1+n:-18].min(1)))-1
    return voltage.clip(0),capacity.clip(0)


def penalty(plant,y):
    v,s=components(plant,y)
    return .1*(v/.003+s/.01)


def proposal_gradient(plant,x,truth):
    # Columns reflect the actual actuator model (including its disk saturation).
    # Central differences of the physical residual; use the active subgradient at
    # the base point rather than differencing the hinge across a change of sign.
    h=2e-5
    basis=h*np.eye(54)
    probes=np.vstack((x+basis,x-basis))
    nv=2*plant.m.original_n
    grad=np.zeros(54)
    ys=base.qp.ac(plant,x,truth)
    for pv,y in zip(truth,ys):
        vrow=1+int(np.argmin(y[1:1+nv]))
        srow=1+nv+int(np.argmin(y[1+nv:-18]))
        util=np.sqrt(max(0.,1-y[srow]))
        if y[vrow]>=0 and util<=1:
            continue
        z=base.qp.ac(plant,probes,pv)
        columns=z[:,[vrow,srow]]
        jac=(columns[:54]-columns[54:])/(2*h)
        if y[vrow]<0:
            grad-=.1/.003*jac[:,0]
        if util>1:
            grad-=.1/.01*jac[:,1]/(2*util)
    return grad/len(truth),penalty(plant,ys).mean()


def diagnostics(net,fore,data,step):
    rows=[]
    for i in data.val:
        p,h,w,truth=base.inputs(data,i)
        last=data.d["old"][i]
        with torch.no_grad():
            x=prior.disk(net.suggest(p,last,h,fore(tensor(h[None]))).numpy())[0]
        _,kw,risk,ok=base.direct.outcomes(p,last,x[None],w,truth)
        applied=x if ok[0] else last
        y=base.qp.ac(p,applied,truth[1:])
        v,s=components(p,y)
        rows.append((step,i,int(ok[0]),kw[0],*risk[0],int((s>1e-7).sum()),
                     float(penalty(p,base.qp.ac(p,x,truth[1:])).mean())))
    return rows


def train(arm,pref,seed):
    w=WEIGHTS[pref]
    tag=f"{arm}_{pref}_{seed}"
    data=base.study_data()
    net,fore=base.networks(seed,data.model)
    rng=np.random.default_rng(seed)
    opt=torch.optim.Adam([{"params":net.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    history=[];updates=[];audit=[];details=[]
    heat=np.zeros((5,6));heat_peak=np.zeros((5,6))
    best={}
    for step in range(41):
        if step%10==0:
            rows=diagnostics(net,fore,data,step)
            details.extend(rows)
            values=np.array(rows).mean(0)
            kw=values[3];risks=values[4:8]
            if step==0:
                caps=risks.copy()
            excess=np.maximum(0.,risks-caps-1e-8)@np.array([1,50,2,20])
            trade=kw/100+w*(risks@RISK)
            history.append((step,kw,*risks,values[2],excess,trade))
            for selector,key in [("strict",(excess,kw)),("trade",(trade,))]:
                if selector not in best or key<best[selector]:
                    best[selector]=key
                    pathfile=OUT/f"{arm}_{pref}_{selector}_{seed}.pt"
                    tmp=pathfile.with_suffix(".part.pt")
                    torch.save(dict(actor=net.state_dict(),fore=fore.state_dict(),selected=step),tmp)
                    tmp.replace(pathfile)
        if step==40:
            break
        if step%prior.FRAGMENT==0:
            starts=rng.choice(data.starts,prior.STREAMS,replace=False)+rng.integers(0,121-prior.FRAGMENT,prior.STREAMS)
            carried=[data.d["old"][i].copy() for i in starts]
        terms=[]; propterms=[]
        opt.zero_grad()
        for j,start in enumerate(starts):
            i=int(start+step%prior.FRAGMENT)
            p,h,weather,truth=base.inputs(data,i)
            np.testing.assert_allclose(truth[1:],base.joint.phase_power(data.d["target"][i][:4]),atol=1e-6)
            last=carried[j]
            path=fore(tensor(h[None]))
            cmd=net.suggest(p,last,h,path)
            x=prior.disk(cmd.detach().numpy())[0]
            dirs=rng.choice([-1.,1.],(prior.DIRECTIONS,54))
            probes=prior.disk(np.vstack((x,x+prior.EPSILON*dirs,x-prior.EPSILON*dirs)))
            end,kw,risk,_=base.direct.outcomes(p,last,probes,weather,truth)
            vals=kw/100+w*(risk@RISK)
            grad=(vals[1:1+len(dirs)]-vals[1+len(dirs):])@dirs/(2*prior.EPSILON*len(dirs))
            terms.append((cmd[0]*tensor(grad)).sum())
            pg,loss=proposal_gradient(p,x,truth[1:])
            routed=(cmd[0]*tensor(w*pg)).sum()
            sens=torch.autograd.grad(routed,path,retain_graph=True)[0].detach().numpy()[0]/w
            heat+=abs(sens);heat_peak=np.maximum(heat_peak,abs(sens))
            if arm=="B":
                routed=(net.suggest(p,last,h,path.detach())[0]*tensor(w*pg)).sum()
            if arm!="A":
                propterms.append(routed)
            if abs(sens).max()>1e-8 and len(audit)<5:
                idx=np.unravel_index(np.argmax(abs(sens)),sens.shape)
                path0=path.detach().clone()
                shifted=[]
                e=3e-4
                # Use centered forecast perturbations only in the physical interval.
                if e<float(path0[(0,*idx)])<1-e:
                    with torch.no_grad():
                        for sign in (1,-1):
                            z=path0.clone();z[(0,*idx)]+=sign*e
                            act=prior.disk(net.suggest(p,last,h,z).numpy())[0]
                            shifted.append(penalty(p,base.qp.ac(p,act,truth[1:])).mean())
                    audit.append((step,i,*idx,sens[idx],(shifted[0]-shifted[1])/(2*e)))
            carried[j]=end[0]
            updates.append((step,i,loss,np.linalg.norm(pg),np.linalg.norm(sens)))
        total=torch.stack(terms).mean()
        if propterms:
            total=total+torch.stack(propterms).mean()
        ids=rng.choice(data.train,128,replace=False)
        mse=(fore(tensor(data.d["past"][ids]))-tensor(data.d["target"][ids])).square().mean()
        (total+10*mse).backward()
        for mod in (net,fore):
            torch.nn.utils.clip_grad_norm_(mod.parameters(),3.,error_if_nonfinite=True)
        opt.step()
    def save(suffix,rows,cols):
        pd.DataFrame(rows,columns=cols.split()).to_csv(OUT/f"{tag}_{suffix}.csv",index=False)
    save("val",history,"step kw vf vp of op accepted excess trade")
    save("detail",details,"step case accepted kw vf vp of op overload_seconds_after_delay proposal_loss")
    save("directions",audit,"step case horizon0 site0 predicted finite_difference")
    save("heat",[(s+1,k+1,heat[s,k]/len(updates),heat_peak[s,k]) for s,k in np.ndindex(5,6)],"second site mean_abs max_abs")
    save("updates",updates,"step case proposal_penalty action_grad_norm potential_forecast_grad_norm")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("arm",choices=list("ABC"));p.add_argument("pref",choices=WEIGHTS);p.add_argument("seed",type=int)
    a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        train(a.arm,a.pref,a.seed)
