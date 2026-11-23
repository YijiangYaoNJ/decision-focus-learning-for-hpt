"""Six HPTs: a single neural command, released only after a feasibility check."""
import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from torch import nn

import ieee123_chen_actors as prior
import ieee123_chen_replay as replay
import ieee123_matched_holdout as windows

OUT = windows.ROOT / "runs/ieee123_direct_v1"


class DirectActor(prior.DispatchActor):
    def __init__(self, seed, load_count):
        _, previous = prior.joint.load_models(seed, prior.joint.OUT/f"init_{seed}.pt")
        super().__init__(seed, previous.state_dict())
        first = self.base.body[0]
        extra = 2 * load_count
        expanded = nn.Linear(first.in_features + extra, first.out_features)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :first.in_features-1].copy_(first.weight[:, :-1])
            expanded.weight[:, -1].copy_(first.weight[:, -1])
            expanded.bias.copy_(first.bias)
        self.base.body[0] = expanded
        self.base.mean = torch.cat((self.base.mean, torch.zeros(extra)))
        self.base.scale = torch.cat((self.base.scale, torch.ones(extra)))

    def command(self, plant, last, history, forecast):
        load = plant.m.series[plant.slot]
        connections = np.column_stack((load.real,load.imag)).ravel()/100
        x = np.r_[prior.features(plant.m,plant.slot,plant.taps,history), connections]
        local = prior.local_features(plant,last,history)
        return self(*(prior.joint.tensor(v[None]) for v in (x,history,forecast,local)))


def accepted_commands(plant, commands, weather):
    # scores applies the established AC capability tests; only pass/fail is used.
    valid = np.isfinite(commands).all(axis=1) & (np.abs(commands)<=1).all(axis=1)
    accepted = np.zeros(len(commands),dtype=bool)
    if valid.any():
        try:
            checked = prior.scores(plant,commands[valid],weather)
            accepted[valid] = [r is not None for r in checked]
        except ArithmeticError:
            for i in np.flatnonzero(valid):
                try:
                    accepted[i] = prior.scores(plant,commands[i:i+1],weather)[0] is not None
                except ArithmeticError:
                    pass
    return accepted


def outcomes(plant,last,commands,weather,truth,*,first_second_held=True,last_frame_plant=None):
    accepted = accepted_commands(plant,commands,weather)
    end = np.where(accepted[:,None],commands,last)
    # One-second actuation lag in offline training, measured release times in replay.
    unique, indices = np.unique(end,axis=0,return_inverse=True)
    same_slot=truth if last_frame_plant is None else truth[:-1]
    seconds=len(same_slot)
    y = prior.eval_rows(plant,np.repeat(unique,seconds,axis=0),np.tile(same_slot,(len(unique),1)))
    frames=y.reshape(len(unique),seconds,-1)
    if last_frame_plant is not None:
        tail=prior.eval_rows(last_frame_plant,unique,np.repeat(truth[-1:],len(unique),axis=0))
        frames=np.concatenate((frames,tail[:,None,:]),axis=1)
    actual=frames[indices]
    if first_second_held:
        actual[:,0] = prior.eval_rows(plant,last[None],truth[:1])[0]
    n = 2*plant.m.original_n
    gap = np.maximum(0,-actual[:,:,1:1+n].min(2))
    util = np.sqrt(np.maximum(0,1-actual[:,:,1+n:-18].min(2)))
    risk = np.column_stack(((gap>0).mean(1),gap.max(1),(util>1+1e-7).mean(1),
                            (util-1).clip(0).max(1)))
    return end,actual[:,:,0].mean(1),risk,accepted


def example(net,data,i,last):
    plant = data.plant(i)
    past = data.d["past"][i]
    forecast = data.ref[i]
    cmd = net.command(plant,last,past,forecast)
    weather = prior.joint.phase_power(np.vstack((past[-1],forecast)))
    truth = prior.joint.phase_power(data.d["truth"][i])
    return cmd,plant,weather,truth


@torch.no_grad()
def validation(net,data):
    rows=[]
    for i in data.val:
        last=data.d["old"][i]
        cmd,plant,weather,truth=example(net,data,i,last)
        _,kw,risk,accepted=outcomes(plant,last,prior.disk(cmd.numpy()),weather,truth)
        rows.append([kw[0],*risk[0],accepted[0]])
    return np.mean(rows,axis=0)


def train(data,seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=DirectActor(seed,len(data.model.lp))
    opt=torch.optim.Adam(net.parameters(),lr=.0005)
    history=[]
    best=None
    for step in range(prior.STEPS+1):
        if step%prior.EVERY==0:
            kw,v,g,o,p,accepted=validation(net,data)
            risks=np.array([v,g,o,p])
            if step==0:
                caps=risks.copy()
            key=(float(np.maximum(0,risks-caps-1e-8)@[1,50,2,20]),kw)
            history.append([step,*key,v,g,o,p,accepted])
            if best is None or key<best:
                best,picked=key,step
                torch.save(net.state_dict(),OUT/f"direct_{seed}.pt")
            if step==prior.STEPS or step-picked>=prior.PATIENCE:
                break
        if step%prior.FRAGMENT==0:
            starts=rng.choice(data.starts,prior.STREAMS,replace=False)+rng.integers(
                0,121-prior.FRAGMENT,prior.STREAMS)
            carried=[data.d["old"][i].copy() for i in starts]
        terms=[]
        for j,start in enumerate(starts):
            i=int(start+step%prior.FRAGMENT)
            directions=rng.choice([-1.,1.],(prior.DIRECTIONS,54))
            last=carried[j]
            cmd,plant,weather,truth=example(net,data,i,last)
            raw=prior.disk(cmd.detach().numpy())[0]
            delta=prior.EPSILON*directions
            probes=prior.disk(np.vstack((raw,raw+delta,raw-delta)))
            end,kw,risk,_=outcomes(plant,last,probes,weather,truth)
            values=prior.op_values(kw,risk)
            slope=(values[1:1+prior.DIRECTIONS]-values[1+prior.DIRECTIONS:])/(2*prior.EPSILON)
            terms.append((cmd[0]*prior.joint.tensor(slope@directions/prior.DIRECTIONS)).sum())
            carried[j]=end[0]
        opt.zero_grad()
        torch.stack(terms).mean().backward()
        nn.utils.clip_grad_norm_(net.parameters(),3.,error_if_nonfinite=True)
        opt.step()
    pd.DataFrame(history,columns="step risk_excess loss_kw voltage_fraction voltage_peak overload_fraction overload_peak accepted".split()).to_csv(OUT/f"validation_{seed}.csv",index=False)
    print("selected",seed,picked,"ended",step,flush=True)


@torch.inference_mode()
def issue(net,plant,last,past,predictor):
    start=perf_counter()
    try:
        forecast=predictor(prior.joint.tensor(past[None])).numpy()[0]
        command=prior.disk(net.command(plant,last,past,forecast).numpy())[0]
        weather=prior.joint.phase_power(np.vstack((past[-1],forecast)))
        accepted=accepted_commands(plant,command[None],weather)[0]
        status="issued" if accepted else "rejected"
    except ArithmeticError:
        command=None
        status="solver_error"
    elapsed=perf_counter()-start
    if elapsed>=5:
        status="timeout"
    return command if status=="issued" else None, elapsed, status


def evaluate(seeds):
    # Reuse the locked weather and initial plant states, independently of legacy actors.
    manifest_path=windows.OUT/"windows.csv"
    archive_path=windows.OUT/"windows.npz"
    lock=json.loads((windows.OUT/"lock.json").read_text())
    if prior.joint.sha(manifest_path)!=lock["manifest"] or prior.joint.sha(archive_path)!=lock["windows"]:
        raise ValueError("August evaluation windows do not match their recorded hashes")
    manifest=pd.read_csv(manifest_path)
    _,model,*_=replay.build()
    forecast=replay.forecast_net(windows.REF)
    actors={}
    for seed in seeds:
        actor=DirectActor(seed,len(model.lp))
        actor.load_state_dict(torch.load(OUT/f"direct_{seed}.pt",weights_only=True))
        actors[seed]=actor.eval()
    rows=[]
    with np.load(archive_path) as archive:
        for row in manifest.itertuples():
            seq=archive[f"{row.date}_pv"]
            truth=prior.joint.phase_power(seq[10:610])
            plant=replay.BatchAC(model,int(row.slot),tuple(archive[f"{row.date}_taps"]))
            for seed,actor in actors.items():
                last=archive[f"{row.date}_initial"].copy()
                last_v=None
                readings=[]
                statuses=[]
                times=[]
                issue(actor,plant,last,np.tile(seq[0],(11,1)),forecast)
                for t in range(0,600,5):
                    new,elapsed,status=issue(actor,plant,last,seq[t:t+11],forecast)
                    times.append(elapsed*1000)
                    statuses.append(status)
                    for s in range(5):
                        applied=new if new is not None and s>=elapsed else last
                        try:
                            state=model.solve(plant.slot,applied,"high",plant.taps,generation_kw=truth[t+s],tolerance=1e-8,max_iterations=250)
                        except ArithmeticError:
                            state=model.solve(plant.slot,applied,"high",plant.taps,generation_kw=truth[t+s],start=last_v,tolerance=1e-8,max_iterations=1500)
                        last_v=state["v"]
                        m=prior.metrics(model,state,plant.slot,"high","direct",model.ratings.ravel())
                        readings.append((m["loss_kw"],m["excursion"],m["overload"]))
                    if new is not None:
                        last=new
                kw,gap,over=np.array(readings).T
                rows.append(dict(date=row.date,seed=seed,kwh=kw.sum()/3600,voltage_s=(gap>0).sum(),
                                 voltage_peak=gap.max(),overload_s=over.sum(),issued=statuses.count("issued"),
                                 rejected=statuses.count("rejected"),timeout=statuses.count("timeout"),
                                 solver_error=statuses.count("solver_error"),ms_p50=np.median(times),
                                 ms_p95=np.quantile(times,.95)))
            print("evaluated",row.date,flush=True)
    table=pd.DataFrame(rows)
    filename="replay.csv" if sorted(seeds)==sorted(windows.SEEDS) else "replay_"+"_".join(map(str,seeds))+".csv"
    table.to_csv(OUT/filename,index=False)
    print(table.groupby("date").mean().mean().to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("train","evaluate","all"))
    parser.add_argument("--seeds",type=int,nargs="+",default=windows.SEEDS)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):
        parser.error("Duplicate seeds")
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.stage in ("train","all"):
            data=prior.April()
            for seed in args.seeds:
                train(data,seed)
        if args.stage in ("evaluate","all"):
            evaluate(args.seeds)
