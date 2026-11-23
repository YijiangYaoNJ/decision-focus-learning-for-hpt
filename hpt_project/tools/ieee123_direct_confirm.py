"""Paired forecast-gradient ablation, single-command issue, and two classical controls."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import ieee123_direct as direct
import ieee123_chen_projection as qp
from ieee123_direct_confirm_data import OUT, plant_model
from ieee123_ramp_dispatch import Snapshot

prior = direct.prior
joint = prior.joint
ARMS = ("frozen", "mse", "joint", "qp", "acopf")
SEEDS = (11,23,37)


class DirectPolicy(direct.DirectActor):
    def suggest(self, plant, last, history, forecast):
        load = plant.m.series[plant.slot]
        x = np.r_[prior.features(plant.m,plant.slot,plant.taps,history),
                  np.column_stack((load.real,load.imag)).ravel()/100]
        local = prior.local_features(plant,last,history)
        return self(joint.tensor(x[None]),joint.tensor(history[None]),forecast,joint.tensor(local[None]))


def networks(seed, model):
    torch.manual_seed(seed)
    actor = DirectPolicy(seed,len(model.lp))
    fore = direct.replay.forecast_net(direct.windows.REF)
    return actor, fore


def study_data():
    data = prior.April.__new__(prior.April)
    prior.verify_file("cases.npz")
    data.d = dict(np.load(prior.APRIL/"cases.npz"))
    valid = np.isfinite(data.d["target"]).all(axis=(1,2))
    data.val = np.flatnonzero(valid & (data.d["date"]>20100422))
    data.train = np.flatnonzero(valid & (data.d["date"]<=20100422))
    data.starts = np.arange(len(valid)-len(data.d["applied"]),len(valid),120)
    assert len(data.val)==36 and len(data.train)==5254
    assert all(np.all(np.isin(np.arange(i,i+120),data.train)) for i in data.starts)
    data.model, data.plants = plant_model(), {}
    with torch.no_grad():
        data.ref=direct.replay.forecast_net(direct.windows.REF)(joint.tensor(data.d["past"])).numpy()
    return data


def inputs(data,i):
    history=data.d["past"][i]
    weather=joint.phase_power(np.vstack((history[-1],data.ref[i])))
    return data.plant(i),history,weather,joint.phase_power(data.d["truth"][i])


@torch.no_grad()
def validation(net,fore,data):
    rows=[]
    for i in data.val:
        plant,h,weather,truth=inputs(data,i)
        last=data.d["old"][i]
        cmd=net.suggest(plant,last,h,fore(joint.tensor(h[None])))
        _,kw,risk,issued=direct.outcomes(plant,last,prior.disk(cmd.numpy()),weather,truth)
        rows.append([kw[0],*risk[0],issued[0]])
    return np.mean(rows,axis=0)


def train(arm,seed):
    data=study_data()
    net,fore=networks(seed,data.model)
    fore.requires_grad_(arm!="frozen")
    opt=torch.optim.Adam([{"params":net.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    rng=np.random.default_rng(seed)
    history,updates=[],[]
    best=None
    checkpoint=OUT/f"{arm}_{seed}.pt"
    for step in range(prior.STEPS+1):
        if step%prior.EVERY==0:
            kw,*rest=validation(net,fore,data)
            risks=np.asarray(rest[:4])
            if step==0:
                caps=risks.copy()
            key=(float(np.maximum(0,risks-caps-1e-8)@[1,50,2,20]),kw)
            history.append((step,*key,*rest))
            if best is None or key<best:
                best,picked=key,step
                tmp=checkpoint.with_suffix(".part.pt")
                torch.save(dict(actor=net.state_dict(),fore=fore.state_dict(),selected=step),tmp)
                tmp.replace(checkpoint)
            if step-picked>=prior.PATIENCE or step==prior.STEPS:
                break
        if step%prior.FRAGMENT==0:
            starts=rng.choice(data.starts,prior.STREAMS,replace=False)+rng.integers(0,121-prior.FRAGMENT,prior.STREAMS)
            carried=[data.d["old"][i].copy() for i in starts]
        opt.zero_grad()
        terms=[]
        for j,start in enumerate(starts):
            i=int(start+step%prior.FRAGMENT)
            plant,h,weather,truth=inputs(data,i)
            path=fore(joint.tensor(h[None]))
            cmd=net.suggest(plant,carried[j],h,path if arm=="joint" else path.detach())
            raw=prior.disk(cmd.detach().numpy())[0]
            dirs=rng.choice([-1.,1.],(prior.DIRECTIONS,54))
            delta=prior.EPSILON*dirs
            probes=prior.disk(np.vstack((raw,raw+delta,raw-delta)))
            end,kw,risk,_=direct.outcomes(plant,carried[j],probes,weather,truth)
            score=prior.op_values(kw,risk)
            slope=(score[1:1+len(dirs)]-score[1+len(dirs):])/(2*prior.EPSILON)
            terms.append((cmd[0]*joint.tensor(slope@dirs/len(dirs))).sum())
            carried[j]=end[0]
        torch.stack(terms).mean().backward()
        task_grad=sum(p.grad.square().sum().item() for p in fore.parameters() if p.grad is not None)**.5
        ids=rng.choice(data.train,128,replace=False)
        mse=(fore(joint.tensor(data.d["past"][ids]))-joint.tensor(data.d["target"][ids])).square().mean()
        if arm!="frozen":
            (10*mse).backward()
        for module in (fore,net):
            torch.nn.utils.clip_grad_norm_(module.parameters(),3.,error_if_nonfinite=True)
        opt.step()
        updates.append((step+1,task_grad,mse.item()))
    assert (max(r[1] for r in updates)>0)==(arm=="joint")
    pd.DataFrame(history,columns="step risk_excess kw voltage_fraction voltage_peak overload_fraction overload_peak accepted".split()).to_csv(OUT/f"{arm}_{seed}_val.csv",index=False)
    pd.DataFrame(updates,columns=["step","forecast_task_grad","mse"]).to_csv(OUT/f"{arm}_{seed}_updates.csv",index=False)
    print(arm,seed,"selected",picked,"stopped",step,flush=True)


class Deadline(Exception):
    pass


def snapshot_opf(plant,last,weather,begin):
    # Exact nonlinear AC constraints at the forecast mean. A common independent
    # six-frame AC screen decides whether the single final command is issued.
    pv=weather.mean(axis=0)
    snap=Snapshot(plant.m,plant.slot,plant.taps,pv)
    jx=jac=None
    def values(x):
        if perf_counter()-begin>=5:
            raise Deadline()
        return snap.at(x)
    def derivative(x):
        nonlocal jx,jac
        values(x)
        if jx is None or not np.array_equal(x,jx):
            jac=plant.jac(x,pv,snap.last_v)
            jx=x.copy()
        return jac
    result=minimize(lambda x:values(x)[0],last,jac=lambda x:derivative(x)[0],
                    method="SLSQP",bounds=[(-1,1)]*54,
                    constraints=[dict(type="ineq",fun=lambda x:values(x)[1:]-2e-5,
                                      jac=lambda x:derivative(x)[1:])],
                    options=dict(ftol=1e-4,maxiter=45))
    return result.x,str(result.status)


@torch.inference_mode()
def issue(arm,actor,fore,ref,plant,last,h,*,trace=None):
    begin=perf_counter()
    solver=""
    cmd=path=None
    try:
        ht=joint.tensor(h[None])
        reference=ref(ht)
        path=reference if fore is ref else fore(ht)
        weather=joint.phase_power(np.vstack((h[-1],reference.numpy()[0])))
        if arm in ("qp","qp_archive_initial","qp_archive_fitted"):
            # Same warm start in both argument positions: no alternative command.
            cmd=qp.local_qp(plant,last,last,h,path.numpy()[0])
        elif arm=="acopf":
            cmd,solver=snapshot_opf(plant,last,weather,begin)
        else:
            cmd=prior.disk(actor.suggest(plant,last,h,path).numpy())[0]
        ok=direct.accepted_commands(plant,cmd[None],weather)[0]
        status="issued" if ok else "rejected"
    except Deadline:
        status="timeout"
    except ArithmeticError:
        status="solver_error"
    elapsed=perf_counter()-begin
    if elapsed>=5:
        status="timeout"
    if trace is not None:
        trace.append((path.numpy()[0].copy() if path is not None else np.full((5,6),np.nan),
                      cmd.copy() if cmd is not None else np.full(54,np.nan)))
    return cmd if status=="issued" else None,status,elapsed,solver


def replay(arm,seed,date,*,directory=OUT,save_actions=False,loss_components=False):
    file=directory/f"d{date}_{arm}_{seed}.csv"
    model=plant_model()
    lock=json.loads((directory/"lock.json").read_text())
    assert date in lock["dates"]
    for name,key in [("windows.npz","windows"),("windows.csv","manifest")]:
        assert joint.sha(directory/name)==lock[key]
    row=pd.read_csv(directory/"windows.csv").set_index("date").loc[date]
    with np.load(directory/"windows.npz") as z:
        seq=z[f"{date}_pv"]; last=z[f"{date}_initial"]
        plant=joint.BatchAC(model,int(row.slot),tuple(z[f"{date}_taps"]))
    ref=direct.replay.forecast_net(direct.windows.REF)
    actor=None
    fore=ref
    if arm in ("qp_archive_initial","qp_archive_fitted"):
        fore=direct.replay.forecast_net(directory/f"{arm}_{seed}.pt")
        if arm=="qp_archive_initial" and seed==11:
            assert all(torch.equal(v,ref.state_dict()[k]) for k,v in fore.state_dict().items())
            fore=ref
    elif arm not in ("qp","acopf"):
        actor,fore=networks(seed,model)
        state=torch.load(directory/f"{arm}_{seed}.pt",weights_only=True)
        actor.load_state_dict(state["actor"]); actor.eval()
        fore.load_state_dict(state["fore"]); fore.eval()
        if arm=="frozen" or state["selected"]==0 or state.get("forecast_is_reference", False):
            fore=ref
    issue(arm,actor,fore,ref,plant,last,np.tile(seq[0],(11,1)))
    rows=[]
    trace=[] if save_actions else None
    applied=[]
    last_v=None
    truth=joint.phase_power(seq[10:610])
    for t in range(0,600,5):
        new,status,elapsed,solver=issue(arm,actor,fore,ref,plant,last,seq[t:t+11],trace=trace)
        vals=[]
        network_kw=[]
        for s in range(5):
            a=new if new is not None and s>=elapsed else last
            if save_actions:
                applied.append(a.copy())
            try:
                state=model.solve(plant.slot,a,"high",plant.taps,generation_kw=truth[t+s],tolerance=1e-8,max_iterations=250)
            except ArithmeticError:
                state=model.solve(plant.slot,a,"high",plant.taps,generation_kw=truth[t+s],start=last_v,tolerance=1e-8,max_iterations=1500)
            last_v=state["v"]
            m=prior.metrics(model,state,plant.slot,"high",arm,model.ratings.ravel())
            if loss_components:
                network_kw.append(m["network_kw"])
            vals.append([m["loss_kw"],m["excursion"],max(m["main_util"],m["port_util"])])
        if new is not None:
            last=new
        kw,v,u=np.array(vals).T
        rows.append((date,arm,seed,t,kw.sum()/3600,(v>0).sum(),v.max(),(u>1+1e-7).sum(),
                     max(0.,u.max()-1),elapsed*1000,status,solver))
        if loss_components:
            network_energy=sum(network_kw)/3600
            rows[-1]+= (network_energy,kw.sum()/3600-network_energy)
    tmp=file.with_suffix(".part")
    if save_actions:
        forecasts,commands=zip(*trace)
        with file.with_suffix(".npz.part").open("wb") as handle:
            np.savez_compressed(handle,forecast=forecasts,proposal=commands,applied=applied)
        file.with_suffix(".npz.part").replace(file.with_suffix(".npz"))
    columns="date arm seed second kwh voltage_s voltage_peak overload_s overload_peak ms status solver".split()
    if loss_components:
        columns.extend(["network_kwh","converter_kwh"])
    pd.DataFrame(rows,columns=columns).to_csv(tmp,index=False)
    tmp.replace(file)
    print("replayed",date,arm,seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=["train","replay"])
    p.add_argument("arm",choices=ARMS);p.add_argument("seed",type=int);p.add_argument("date",nargs="?",type=int)
    args=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.stage=="train":
            train(args.arm,args.seed)
        else:
            replay(args.arm,args.seed,args.date)
