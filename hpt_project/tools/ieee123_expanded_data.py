"""Versioned continuous trajectories for the six-HPT predictor and closed-loop dispatcher."""
import argparse
import json
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import ieee123_hourly_information as hourly
from ieee123_fast_ramps import predictor
from ieee123_ramp_windows import ARCHIVE, SIX, day_data

base=hourly.base
RUNS=hourly.OUT.parent
OUT=RUNS/"ieee123_expanded_trajectories_v1"
RAW=RUNS/"ieee123_joint5_trajectories_v1"/"cases.npz"
LENGTH=5400
CYCLES=LENGTH//5
SEED=20261122
PERIODS={"morning":(7,10.5),"midday":(10.5,14),"afternoon":(14,17.5)}
MANIFESTS=("windows.csv","paired_windows.csv","manifest.csv","by_date.csv","by_date_seed.csv")


def previous_dates():
    records=[]
    for name in MANIFESTS:
        for p in sorted(RUNS.glob(f"ieee123*/**/{name}")):
            if OUT in p.parents: continue
            df=pd.read_csv(p)
            if "date" not in df: continue
            if "split" in df:
                df=df[df.split!="train"]
            dates=df.date.astype(str).str.replace("-","",regex=False)
            for day in dates[dates.str.fullmatch(r"2010\d{4}")].unique():
                records.append((int(day),p.relative_to(RUNS).as_posix()))
    return pd.DataFrame(records,columns=["date","prior_manifest"]).drop_duplicates()


def assign(date,seen):
    if 20100401<=date<=20100422 and date not in seen: return "train"
    if 20100501<=date<=20100516 and date not in seen: return "train"
    if 20100517<=date<=20100523 and date not in seen: return "validation"
    if 20100524<=date<=20100531 and date not in seen: return "test_may"
    if 20100901<=date<=20100930: return "test_september"
    return None


def candidates(pv,low,high,blocked):
    valid=[]
    for t in range(int(low*3600),int(high*3600)-LENGTH+1,900):
        if any(t-3600<=b and t+LENGTH+1>=a for a,b in blocked): continue
        if np.isfinite(pv[t-3600:t]).all() and np.isfinite(pv[t-10:t+LENGTH+1,SIX]).all():
            valid.append(t)
    return valid


def make_index(pv,slot0):
    samples=np.lib.stride_tricks.sliding_window_view(pv,16,axis=0)[::5].swapaxes(1,2).copy()
    assert samples.shape==(CYCLES,16,6)
    return dict(past=samples[:,:11],truth=samples[:,10:15],target=samples[:,11:16],
                slot=slot0+np.arange(CYCLES)*5//900)


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/"lock.json").exists(): return
    seen=previous_dates()
    seen.to_csv(OUT/"prior_evaluation_dates.csv",index=False)
    old=base.study_data()
    legacy=pd.DataFrame({"date":old.d["date"][old.train],"second":old.d["second"][old.train]})
    sep=pd.read_csv(hourly.study.CONF/"windows.csv").set_index("date")
    rows=[]; inventory=[]; arrays={}; seeds=[]; ramp_rows=[]; digests={}
    for month in (4,5,9):
        source=ARCHIVE.with_name(f"oahu_2010{month:02}.zip")
        digests[source.name]=base.joint.sha(source)
        with zipfile.ZipFile(source) as archive:
            for file in sorted(archive.namelist()):
                date=int(file[:8]); split=assign(date,set(seen.date))
                if split is None: continue
                pv=day_data(archive,file)
                blocked=[]
                if date in sep.index:
                    t=int(sep.loc[date,"second"])
                    blocked=[(t-3600,t+601)]
                for low_t in legacy.loc[legacy.date==date,"second"].to_numpy():
                    blocked.append((int(low_t)-10,int(low_t)+5))
                for b,(period,(low,high)) in enumerate(PERIODS.items()):
                    valid=candidates(pv,low,high,blocked)
                    inventory.append((date,split,period,len(valid)))
                    if not valid: continue
                    t=int(np.random.default_rng([SEED,date,b]).choice(valid))
                    slot=(datetime.strptime(str(date),"%Y%m%d").timetuple().tm_yday-1)*96+t//900
                    uid=f"{date}_{t}"; seq=pv[t-10:t+LENGTH+1,SIX]
                    idx=make_index(seq,slot)
                    ramp=idx["target"][:,-1]-idx["past"][:,-1]
                    spread=ramp.max(axis=1)-ramp.min(axis=1)
                    ramp_rows.append((uid,int((spread>=.05).sum()),int(((ramp.max(1)>.025)&(ramp.min(1)<-.025)).sum()),
                                      int((abs(ramp).max(1)<.01).sum())))
                    h=pv[t-3600:t].reshape(4,900,9).mean(1)[::-1].ravel()
                    a=2*np.pi*slot/96
                    seeds.append(np.r_[h,np.sin(a),np.cos(a)])
                    arrays[uid+"_pv"]=seq
                    rows.append((uid,date,t,slot,split,period))
    sys.path.insert(0,str(RUNS.parent/"src"))
    predictions,slow_weights=predictor(np.stack(seeds))
    model=old.model
    frame=pd.DataFrame(rows,columns=["id","date","second","slot","split","period"])
    physical=[]
    for r,pred in zip(frame.itertuples(),predictions):
        st=model.step(r.slot,hourly.ZERO,"high",generation_kw=base.joint.phase_power(pred))
        init=hourly.local(st,model,hourly.ZERO)
        taps=np.asarray(st["taps"])
        arrays[r.id+"_initial"]=init; arrays[r.id+"_taps"]=taps
        state=model.solve(r.slot,init,"high",taps,generation_kw=base.joint.phase_power(arrays[r.id+"_pv"][10]),
                          tolerance=1e-8,max_iterations=1500)
        m=base.prior.metrics(model,state,r.slot,"high","dataset",model.ratings.ravel())
        pq=model.series[r.slot:r.slot+6].sum(axis=1)
        physical.append((r.id,min(m["vmin"]-.95,1.05-m["vmax"]),1-max(m["port_util"],m["main_util"]),
                         *[x for v in (pq.real,pq.imag) for x in (v.min(),v.max())]))
    frame=frame.merge(pd.DataFrame(ramp_rows,columns=["id","spatial_cycles","opposing_cycles","steady_cycles"]),on="id")
    frame=frame.merge(pd.DataFrame(physical,columns=["id","initial_v_margin","initial_capacity_margin",
                      "min_load_kw","max_load_kw","min_load_kvar","max_load_kvar"]),on="id")
    frame.to_csv(OUT/"manifest.csv",index=False)
    pd.DataFrame(inventory,columns=["date","split","period","eligible_starts"]).to_csv(OUT/"coverage.csv",index=False)
    members={}
    for split,part in frame.groupby("split"):
        tmp=OUT/f"{split}.part.npz"
        np.savez_compressed(tmp,**{f"{uid}_{k}":arrays[f"{uid}_{k}"] for uid in part.id for k in ("pv","taps","initial")})
        target=OUT/f"{split}.npz"; tmp.replace(target)
        members[target.name]=base.joint.sha(target)
    lock=dict(seed=SEED,seconds=LENGTH,cycle_seconds=5,source=digests,slow_forecaster=slow_weights,
        feeder=json.loads((RUNS/"ieee123_direct_confirm_v1"/"lock.json").read_text())["feeder"],
        legacy_sha=base.joint.sha(RAW),legacy_count=len(legacy),legacy_dates=sorted(map(int,legacy.date.unique())),
        legacy_protocol="Original 5254 train indices only. Previous April validation and test records excluded.",
        extra_protocol="One uniform complete 90-minute interval per available date and time band, no replacement. Whole dates split. No selection by controller outcomes. Test September excludes prior ten-minute window plus historical feature support.",
        files={**members,**{n:base.joint.sha(OUT/n) for n in ("manifest.csv","coverage.csv","prior_evaluation_dates.csv")}})
    temp=OUT/"lock.part";temp.write_text(json.dumps(lock,indent=2));temp.replace(OUT/"lock.json")


class Trajectory:
    """One chronological unit for physics-feedback training, or for evaluation with locked weights."""
    def __init__(self,pv,slot,taps,initial,model,ref_net):
        self.d=make_index(pv,slot)
        self.initial=initial.copy()
        self.taps=tuple(taps)
        self.model=model;self.plants={}
        with torch.inference_mode():
            self.ref=ref_net(base.joint.tensor(self.d["past"].copy())).numpy()

    def plant(self,i):
        slot=int(self.d["slot"][i])
        if slot not in self.plants: self.plants[slot]=base.joint.BatchAC(self.model,slot,self.taps)
        return self.plants[slot]

    def feedback(self,i,last,actor,fore,rng,weight=4.,forecast_feedback=True):
        final=i==len(self.ref)-1
        next_plant=None
        if not final and self.d["slot"][i+1]!=self.d["slot"][i]:
            next_plant=self.plant(i+1)
        return hourly.study.full.sample(self,i,last,actor,fore,rng,"C",weight,forecast_feedback=forecast_feedback,
                                        applied_interval=True,final_cycle=final,last_frame_plant=next_plant)


class ExpandedTrainingSet:
    """Original training pool plus continuous extensions. Never opens validation or test archives."""
    def __init__(self):
        lock=json.loads((OUT/"lock.json").read_text())
        for name in ("manifest.csv","train.npz"):
            if base.joint.sha(OUT/name)!=lock["files"][name]: raise ValueError(name)
        if base.joint.sha(RAW)!=lock["legacy_sha"]: raise ValueError(RAW)
        self.legacy=base.study_data()
        self.rows=pd.read_csv(OUT/"manifest.csv").query("split=='train'").set_index("id")
        self.ref_net=base.direct.replay.forecast_net(base.direct.windows.REF)

    def epoch(self,seed,epoch):
        """Visit every legacy singleton/chain and every new chain once; carry the returned action within each."""
        d=self.legacy
        chains=[np.arange(s,s+120) for s in d.starts]
        single=np.setdiff1d(d.train,np.concatenate(chains))
        units=[("legacy",c) for c in chains]+[("legacy",np.array([i])) for i in single]+[("expanded",k) for k in self.rows.index]
        rng=np.random.default_rng([seed,epoch])
        with np.load(OUT/"train.npz") as z:
            for j in rng.permutation(len(units)):
                kind,key=units[j]
                if kind=="legacy":
                    yield d,key,d.d["old"][key[0]].copy()
                else:
                    r=self.rows.loc[key]
                    data=Trajectory(z[key+"_pv"],int(r.slot),z[key+"_taps"],z[key+"_initial"],d.model,self.ref_net)
                    yield data,range(CYCLES),data.initial.copy()


def evaluation_trajectories(split):
    if split not in ("validation","test_may","test_september"): raise ValueError(split)
    lock=json.loads((OUT/"lock.json").read_text())
    for name in ("manifest.csv",split+".npz"):
        if base.joint.sha(OUT/name)!=lock["files"][name]: raise ValueError(name)
    frame=pd.read_csv(OUT/"manifest.csv")
    model=base.plant_model()
    ref=base.direct.replay.forecast_net(base.direct.windows.REF)
    with np.load(OUT/(split+".npz")) as z:
        for r in frame[frame.split==split].itertuples():
            yield r,Trajectory(z[r.id+"_pv"],r.slot,z[r.id+"_taps"],z[r.id+"_initial"],model,ref)


def audit():
    lock=json.loads((OUT/"lock.json").read_text());f=pd.read_csv(OUT/"manifest.csv")
    for name,digest in lock["files"].items(): assert base.joint.sha(OUT/name)==digest
    groups={k:set(g.date) for k,g in f.groupby("split")}
    groups["train"].update(lock["legacy_dates"])
    assert len(groups)==4 and sum(map(len,groups.values()))==len(set.union(*groups.values()))
    assert max(groups["train"])<min(groups["validation"])<=max(groups["validation"])<min(groups["test_may"])
    seen=set(pd.read_csv(OUT/"prior_evaluation_dates.csv").date)
    assert not seen & groups["test_may"]
    sept=pd.read_csv(hourly.study.CONF/"windows.csv").set_index("date")
    legacy=base.study_data()
    unique=set(zip(legacy.d["date"][legacy.train],legacy.d["second"][legacy.train]))
    assert len(unique)==lock["legacy_count"]
    for split,part in f.groupby("split"):
        with np.load(OUT/f"{split}.npz") as z:
            for r in part.itertuples():
                p=z[r.id+"_pv"]; idx=make_index(p,r.slot)
                assert p.shape==(LENGTH+11,6) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                assert len(np.unique(idx["slot"]))==6 and np.array_equal(idx["target"][:-1,-1],idx["past"][1:,-1])
                assert z[r.id+"_initial"].shape==(54,) and np.isfinite(z[r.id+"_initial"]).all()
                if r.date in sept.index:
                    start=int(sept.loc[r.date,"second"]); assert r.second+LENGTH+1<start-3600 or r.second-3600>start+601
                keys={(r.date,r.second+t) for t in range(0,LENGTH,5)}
                assert not keys & unique
                unique.update(keys)
    rng=np.random.default_rng(182)
    with np.load(OUT/"train.npz") as z:
        r=f.query("split=='train'").iloc[0]; ds=ExpandedTrainingSet()
        tr=Trajectory(z[r.id+"_pv"],int(r.slot),z[r.id+"_taps"],z[r.id+"_initial"],ds.legacy.model,ds.ref_net)
        actor,fore=base.networks(11,tr.model)
        ids=range(2);end=tr.initial;terms=[]
        for i in ids:
            signal,end,_=tr.feedback(i,end,actor,fore,rng)
            terms.append(signal)
        mse=(fore(base.joint.tensor(tr.d["past"][:2].copy()))-base.joint.tensor(tr.d["target"][:2].copy())).square().mean()
        optimizer=torch.optim.Adam([*actor.parameters(),*fore.parameters()],lr=1e-4)
        (torch.stack(terms).mean()+10*mse).backward()
        grad={k:float(torch.nn.utils.clip_grad_norm_(v.parameters(),3.,error_if_nonfinite=True)) for k,v in [("action",actor),("forecast",fore)]}
        assert min(grad.values())>0; optimizer.step()
        i=179
        assert tr.plant(i).slot+1==tr.plant(i+1).slot
        plant,h,weather,_=base.inputs(tr,i)
        truth=base.joint.phase_power(tr.d["target"][i])
        _,loss,_,_=base.direct.outcomes(plant,end,end[None],weather,truth,first_second_held=False,last_frame_plant=tr.plant(i+1))
        expected=np.mean([base.prior.eval_rows(p,end[None],g[None])[0,0] for p,g in zip([plant]*4+[tr.plant(i+1)],truth)])
        np.testing.assert_allclose(loss[0],expected,rtol=1e-8)
    # Exercise the epoch used by training, including the merge with the old pool.
    legacy_indices=set()
    new_slots=set()
    for data,indices,initial in ds.epoch(11,1):
        assert initial.shape==(54,) and np.isfinite(initial).all()
        if data is ds.legacy:
            assert not legacy_indices.intersection(indices)
            legacy_indices.update(indices)
        else:
            start=int(data.d["slot"][0])
            assert start not in new_slots and indices==range(CYCLES)
            new_slots.add(start)
    assert legacy_indices==set(ds.legacy.train)
    assert new_slots==set(ds.rows.slot)
    epoch_cycles=len(legacy_indices)+len(new_slots)*CYCLES
    for split in ("validation","test_may","test_september"):
        r,test_traj=next(evaluation_trajectories(split))
        assert r.date not in groups["train"] and len(test_traj.d["target"])==CYCLES
    tab=f.groupby("split").agg(extended_dates=("date","nunique"),windows=("id","size"),
             spatial_cycles=("spatial_cycles","sum"),opposing_cycles=("opposing_cycles","sum"),steady_cycles=("steady_cycles","sum"))
    tab["hours"]=tab.windows*LENGTH/3600;tab["new_cycles"]=tab.windows*CYCLES
    tab["unique_dates"]=[len(groups[s]) for s in tab.index];tab["total_cycles"]=tab.new_cycles
    tab.loc["train","total_cycles"]+=lock["legacy_count"]
    tab.to_csv(OUT/"counts.csv")
    (OUT/"audit.json").write_text(json.dumps(dict(unique_timestamp_count=len(unique),disjoint_splits=True,
        distinct_legacy_training=lock["legacy_count"],epoch_loader_training_cycles=epoch_cycles,smoke_update_grad_norm=grad,
        cross_slot_error_kw=abs(float(loss[0]-expected)),test_used_for_model_updates=False),indent=2))
    print(tab.to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=["prepare","audit"])
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.stage=="prepare":prepare()
    audit()
