"""Chen–Lin adaptation: a forecast-driven local convex dispatch and an offline realized-scenario projection.

All distances are Euclidean in the existing 54 normalized phase controls. The
full nonlinear AC constraints are re-evaluated after every projection repair.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import ieee123_joint5 as joint
from ieee123_joint5_trajectories import OUT as APRIL, verify_file
from ieee123_interval_ac import BatchAC
from ieee123_interval_data import phase_power
from ieee123_service_areas import build
from ieee123_service_areas import metrics
import numpy as np
import osqp
import pandas as pd
import torch
from scipy.sparse import csc_matrix, eye, vstack
from threadpoolctl import threadpool_limits

OUT = APRIL.with_name("ieee123_chen_three_arm_v1")/"training"
MARGIN = .003
IDENTITY = eye(54, format="csc")


def ac(plant, action, power):
    try:
        return plant.evaluate(action,power)
    except ArithmeticError:
        return plant.evaluate(action,power,max_iterations=1500)


def diagnostics(plant, y):
    nv=2*plant.m.original_n
    voltage=np.maximum(-y[:,1:1+nv].min(1),0.)
    winding=np.sqrt(np.maximum(1-y[:,1+nv:-54].min(1),0.))
    port=np.sqrt(np.maximum(1-y[:,-54:-18].min(1),0.))
    return voltage,np.maximum(winding,port)


def jacobian(plant,x,pv):
    try:
        return plant.jac(x,pv,None)
    except ArithmeticError:
        step=np.where(x>.999,-1e-3,1e-3)
        test=np.vstack((x,x+np.diag(step)))
        y=plant.evaluate(test,pv,max_iterations=1500)
        return ((y[1:]-y[0])/step[:,None]).T


def feasible(plant,x,y):
    nv=2*plant.m.original_n
    return bool(np.max(abs(x))<=1 and y[:,1:1+nv].min()>=0 and
                y[:,1+nv:-54].min()>=1-(1+1e-7)**2 and y[:,-54:].min()>=0)


def disk(x):
    a=np.clip(x,-1,1).reshape(18,3).copy()
    a[:,:2]/=np.maximum(1,np.linalg.norm(a[:,:2],axis=1,keepdims=True))
    return a.ravel()


def step_qp(x, jac, raw, gradient, curvature, radius):
    # Solve the local, strongly convex QP in normalized action increments.
    # A box bound proves rows strictly positive throughout it are redundant.
    keep=raw<=radius*np.abs(jac).sum(1)+1e-7
    active=jac[keep]
    lb=np.maximum(-radius,-1.-x)
    ub=np.minimum(radius,1.-x)
    solver=osqp.OSQP()
    solver.setup(P=IDENTITY*curvature,q=gradient,A=vstack((csc_matrix(active),IDENTITY),format="csc"),
                 l=np.r_[-raw[keep],lb],u=np.r_[np.full(len(active),np.inf),ub],
                 eps_abs=2e-6,eps_rel=2e-6,max_iter=3000,polishing=False,verbose=False)
    sol=solver.solve(raise_error=False)
    if sol.info.status_val not in (1,2):
        return None
    return disk(x+sol.x)


def local_qp(plant,old,fixed,past,prediction):
    weather=np.vstack((np.asarray(past)[-1],np.asarray(prediction).reshape(5,6)))
    power=phase_power(weather)
    current=disk(old)
    fallback=disk(fixed)
    nv=2*plant.m.original_n
    def key(x,y):
        return (not feasible(plant,x,y),max(0.,MARGIN-y[:,1:1+nv].min()),
                max(0.,-y[:,1+nv:].min()),float(y[:,0].mean()))
    y=ac(plant,current,power)
    yf=ac(plant,fallback,power)
    if key(fallback,yf)<key(current,y):
        current,y=fallback,yf
    # Common action derivative at the forecast mean, scenario-wise offsets at
    # all six forecast frames. This is also the exact inference routine for A0.
    jac=jacobian(plant,current,power.mean(0))
    raw=y[:,1:].min(0)
    tightened=raw.copy()
    tightened[:nv]-=MARGIN
    proposal=step_qp(current,jac[1:],tightened,jac[0],20.,.35)
    if proposal is None:
        proposal=step_qp(current,jac[1:],raw,jac[0],20.,.35)
    if proposal is None:
        return current
    candidates=np.stack([disk(current+alpha*(proposal-current)) for alpha in (1.,.5,.25)])
    results=ac(plant,np.repeat(candidates,len(power),axis=0),np.tile(power,(len(candidates),1))).reshape(3,len(power),-1)
    best=current
    best_key=key(current,y)
    for x,yr in zip(candidates,results):
        score=key(x,yr)
        if score<best_key:
            best_key,best=score,x
    return best


def optimize(plant,old,fixed,past,prediction):
    """Produce a proposal using current/past and five forecasts only (no realized future)."""
    return local_qp(plant,old,fixed,past,prediction)


def project(plant,proposal,truth_power):
    """Successive nearest-point QPs. No slack may satisfy the final AC test."""
    x=proposal.copy()
    y=ac(plant,x,truth_power)
    for iteration in range(5):
        if feasible(plant,x,y):
            return x,y,iteration
        if iteration==4:
            break
        jac=jacobian(plant,x,truth_power.mean(0))[1:]
        raw=y[:,1:].min(0)-1e-6
        # Minimize 0.5 || x + dx - proposal ||² under true-weather AC
        # linearizations. Repeat if the full AC replay still has a violation.
        trial=step_qp(x,jac,raw,x-proposal,1.,2.)
        if trial is None:
            break
        x=trial
        y=ac(plant,x,truth_power)
    return None,y,-1


def objective(plant,old,fixed,past,path,truth_power):
    x=optimize(plant,old,fixed,past,path)
    try:
        projected,readings,it=project(plant,x,truth_power)
    except ArithmeticError:
        return np.nan,-2
    if projected is None:
        return np.nan,it
    # Dimensionless realized cost plus mean squared normalized projection distance.
    return float(readings[:,0].mean()/100+np.mean((x-projected)**2)),it


def train(seed):
    verify_file("cases.npz")
    archive=APRIL/"cases.npz"
    d=dict(np.load(archive))
    _,m,*_=build()
    valid=np.isfinite(d["target"]).all((1,2))
    tr=np.flatnonzero(valid&(d["date"]<=20100422))
    va=np.flatnonzero(valid&(d["date"]>20100422))
    assert len(va)==36 and max(d["date"][tr])<min(d["date"][va])
    plants={}
    def case(i):
        key=int(d["slot"][i]),tuple(d["taps"][i])
        if key not in plants:
            plants[key]=BatchAC(m,*key)
        return plants[key]
    fixed=np.load(joint.SOURCE/"windows.npz")["fixed"]
    init=joint.OUT/f"init_{seed}.pt"
    f,_=joint.load_models(seed,init)
    optimizer=torch.optim.AdamW(f.parameters(),lr=.00015,weight_decay=.0001)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    hist=joint.tensor(d["past"]); truth=joint.tensor(d["target"])
    best=None; picked=0; records=[]; updates=[]
    stem=OUT/f"A_{seed}"
    started=perf_counter()
    for step in range(121):
        if step%10==0:
            with torch.no_grad():
                predicted=f(hist[va]).numpy()
            vals=[]
            for i,p in zip(va,predicted):
                plant=case(i);power=phase_power(np.vstack((d["past"][i,-1],d["target"][i])))
                x=optimize(plant,d["old"][i],fixed,d["past"][i],p)
                yr=ac(plant,x,power)
                g,u=diagnostics(plant,yr)
                vals.extend(zip(g,u,yr[:,0]))
            gap,util,loss=np.asarray(vals).T
            key=((util>1+1e-7).mean(),(gap>0).mean(),loss.mean())
            records.append(dict(step=step,over=key[0],voltage=key[1],kw=key[2],peak=gap.max(),
                                mse=float(np.mean((predicted-d["target"][va])**2))))
            if best is None or key<best:
                best=key;picked=step
                torch.save(dict(fore=f.state_dict()),stem.with_suffix(".pt"))
            pd.DataFrame(records).to_csv(stem.with_suffix(".csv"),index=False)
            print("A",seed,step,"best",picked,flush=True)
            if step>=40 and step-picked>=40 or step==120:
                break
        ids=rng.choice(tr,4,replace=False)
        prediction=f(hist[ids])
        forecasts=prediction.detach().numpy().astype(float)
        grad=np.zeros_like(forecasts); valid_pairs=projected=failed=0
        eps=.001
        for j,(i,p) in enumerate(zip(ids,forecasts)):
            pw=phase_power(np.vstack((d["past"][i,-1],d["target"][i])))
            # Simultaneous central perturbation differentiates the *whole*
            # executed solve→projection→cost chain, inclusive of active-set changes.
            direction=rng.choice((-1.,1.),size=p.shape)
            sign=[]
            for h in (1.,-1.):
                obj,it=objective(case(i),d["old"][i],fixed,d["past"][i],np.clip(p+h*eps*direction,0,1),pw)
                projected+=int(it>0);failed+=int(it<0)
                sign.append(obj)
            if np.isfinite(sign).all():
                grad[j]=direction*(sign[0]-sign[1])/(2*eps)
                valid_pairs+=1
        loss_phys=(prediction*joint.tensor(grad)).sum((1,2)).mean()
        mse=(prediction-truth[ids]).square().mean()
        optimizer.zero_grad()
        loss_phys.backward(retain_graph=True)
        gnorm=sum(v.grad.square().sum().item() for v in f.parameters() if v.grad is not None)**.5
        (10*mse).backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(),3.,error_if_nonfinite=True)
        optimizer.step()
        updates.append(dict(step=step+1,valid_pairs=valid_pairs,projected=projected,unresolved=failed,
                            dfl_grad=gnorm,mse=float(mse.detach()),ids=",".join(map(str,ids))))
        pd.DataFrame(updates).to_csv(OUT/f"A_{seed}_updates.csv",index=False)
    final=torch.load(stem.with_suffix(".pt"),weights_only=True)["fore"]
    start=torch.load(init,weights_only=True)["fore"]
    detail=dict(seed=seed,selected=picked,stopped=step,initial_sha=joint.sha(init),selected_sha=joint.sha(stem.with_suffix(".pt")),
                max_parameter_change=max((final[k]-start[k]).abs().max().item() for k in final),
                valid_rows=len(tr),validation_rows=len(va),unique_train=int(len({i for r in updates for i in r["ids"].split(",")})),
                valid_pairs=sum(r["valid_pairs"] for r in updates),projection_nonidentity=sum(r["projected"] for r in updates),
                projection_failed=sum(r["unresolved"] for r in updates),nonzero_dfl_steps=sum(r["dfl_grad"]>1e-12 for r in updates),
                elapsed_s=perf_counter()-started,gradient="one central Rademacher draw / case over 30 predictions; eps=0.001",
                validation="realized cost and risks of unprojected proposal across current plus five frames; no projected oracle onboard")
    stem.with_suffix(".json").write_text(json.dumps(detail,indent=2))
    print(json.dumps(detail),flush=True)


def audit():
    """Check forecast-stage determinism, realized AC projection, and directional differences."""
    verify_file("cases.npz")
    d=dict(np.load(APRIL/"cases.npz"))
    _,m,*_=build()
    seed=11
    f,_=joint.load_models(seed,joint.OUT/f"init_{seed}.pt")
    val=np.flatnonzero(np.isfinite(d["target"]).all((1,2))&(d["date"]>20100422))
    fixed=np.load(joint.SOURCE/"windows.npz")["fixed"]
    rng=np.random.default_rng(92117)
    rows=[]
    for i in val:
        plant=BatchAC(m,int(d["slot"][i]),tuple(d["taps"][i]))
        h=d["past"][i]
        path=f(joint.tensor(h[None])).detach().numpy()[0].astype(float)
        pw=phase_power(np.vstack((h[-1],d["target"][i])))
        x=optimize(plant,d["old"][i],fixed,h,path)
        proj,y,it=project(plant,x,pw)
        row=dict(index=i,iterations=it)
        if proj is not None:
            g,u=diagnostics(plant,y)
            native=[]
            for pv in pw:
                state=m.solve(plant.slot,proj,"high",plant.taps,generation_kw=pv,tolerance=1e-9,max_iterations=1500)
                native.append(metrics(m,state,plant.slot,"high","offline_audit",m.ratings.ravel()))
            row.update(distance_norm=float(np.linalg.norm(x-proj)),max_voltage_gap=max(z["excursion"] for z in native),
                       worst_util=max(max(z["main_util"],z["port_util"]) for z in native),
                       batch_native_energy_error_kw=max(abs(a["loss_kw"]-b) for a,b in zip(native,y[:,0])))
        direction=rng.choice((-1.,1.),size=path.shape)
        for eps,label in ((.001,"full"),(.0005,"half")):
            vals=[objective(plant,d["old"][i],fixed,h,np.clip(path+t*eps*direction,0,1),pw)[0] for t in (1.,-1.)]
            row[label]=(vals[0]-vals[1])/(2*eps)
        rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"A_projection_audit.csv",index=False)
    directions=df.dropna(subset=["full","half"])
    active=directions[(directions.full.abs()>1e-6)|(directions.half.abs()>1e-6)]
    paired=np.minimum(active.full.abs(),active.half.abs())/np.maximum(active.full.abs(),active.half.abs())
    physical=df.dropna(subset=["worst_util"])
    detail=dict(rows=len(df),successful=len(physical),nonidentity=int((df.iterations>0).sum()),
        native_unsafe=int(((physical.worst_util>1+1e-7)|(physical.max_voltage_gap>0)).sum()),
        native_loss_error_kw=physical.batch_native_energy_error_kw.max(),valid_direction_pairs=len(directions),
        nonflat_pairs=len(active),sign_agreement=float((active.full*active.half>0).mean()) if len(active) else None,
        median_min_max_slope_ratio=float(paired.median()) if len(paired) else None,
        note="Two finite differences along each validation state's common random direction. Not an analytical KKT gradient or global projection guarantee.")
    # The 36 isolated validation cases can contain no successfully repaired
    # nonidentity projection. Audit a fixed random sample of training states
    # separately to exercise the curved, active-constraint path as well.
    tr=np.flatnonzero(np.isfinite(d["target"]).all((1,2))&(d["date"]<=20100422))
    repairs=[]
    for i in np.random.default_rng(34076).choice(tr,96,replace=False):
        plant=BatchAC(m,int(d["slot"][i]),tuple(d["taps"][i]))
        h=d["past"][i]; p=f(joint.tensor(h[None])).detach().numpy()[0]
        power=phase_power(np.vstack((h[-1],d["target"][i])))
        x=optimize(plant,d["old"][i],fixed,h,p)
        projected,y,it=project(plant,x,power)
        rec=dict(index=i,iterations=it)
        if it>0:
            native=[]
            for pv in power:
                state=m.solve(plant.slot,projected,"high",plant.taps,generation_kw=pv,tolerance=1e-9,max_iterations=1500)
                native.append(metrics(m,state,plant.slot,"high","repair_audit",m.ratings.ravel()))
            rec.update(max_gap=max(z["excursion"] for z in native),max_util=max(max(z["port_util"],z["main_util"]) for z in native),
                       distance=float(np.linalg.norm(x-projected)))
        repairs.append(rec)
    repair_df=pd.DataFrame(repairs)
    repair_df.to_csv(OUT/"A_active_projection_checks.csv",index=False)
    audited=repair_df[repair_df.iterations>0]
    detail["train_repair_audit"]=dict(cases=len(repairs),repaired=len(audited),unresolved=int((repair_df.iterations<0).sum()),
                                      native_unsafe=int(((audited.max_gap>0)|(audited.max_util>1+1e-7)).sum()) if len(audited) else 0)
    (OUT/"A_projection_audit.json").write_text(json.dumps(detail,indent=2))
    print(json.dumps(detail),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",nargs="+",type=int,default=[11,23,37])
    parser.add_argument("--audit",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.audit:
            audit()
        else:
            for seed in args.seeds:
                train(seed)
