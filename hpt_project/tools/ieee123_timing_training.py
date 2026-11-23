"""Matched timing and forecast-feedback ablations, with closed-loop development selection."""
import hashlib
import itertools
import json
import shutil
import sys

import numpy as np
import pandas as pd
import torch

import ieee123_development as dev

full, base = dev.full, dev.base
OUT = dev.OUT.with_name("ieee123_interval_feedback_v1")
CONF = OUT/"september"
MODES = ("old_frozen","aligned_frozen","aligned_mse","aligned_joint")
JOBS = list(itertools.product(MODES,full.SEEDS))
SELECT = list(range(20100423,20100427))
ENDPOINTS = ["kwh","network_kwh","converter_kwh",*dev.RISK,"issued","timeouts","p50_ms","p95_ms"]
MAX_EPOCHS = 4


def save_json(path,value):
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(value,indent=2),encoding="utf8")
    tmp.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    CONF.mkdir(exist_ok=True)
    if (CONF/"lock.json").exists():
        return
    for name in ("windows.npz","windows.csv","lock.json"):
        shutil.copyfile(dev.OUT/name,OUT/name)
    sys.path.insert(0,str(dev.previous.ROOT/"src"))
    archive=dev.trajectories.ARCHIVE.with_name("oahu_201009.zip")
    draws=dev.trajectories.choose(archive,1,np.random.default_rng(211123))
    assert draws and all(20100901<=r[0]<=20100930 for r in draws)
    forecasts,_=dev.previous.predictor(np.stack([r[3] for r in draws]))
    model=base.plant_model()
    manifest=[]; arrays={}
    for (date,t,slot,_,pv),pred in zip(draws,forecasts):
        assert pv.shape==(611,6)
        state=model.step(slot,dev.previous.ZERO,"high",generation_kw=dev.previous.phase_power(pred))
        arrays.update({f"{date}_pv":pv,f"{date}_initial":dev.previous.local(state,model,dev.previous.ZERO),
                       f"{date}_taps":state["taps"]})
        manifest.append((date,t,slot))
    pd.DataFrame(manifest,columns=["date","second","slot"]).to_csv(CONF/"windows.csv",index=False)
    np.savez_compressed(CONF/"windows.npz",**arrays)
    save_json(OUT/"protocol.json",dict(modes=MODES,seeds=list(full.SEEDS),weight=4.,steps=list(dev.EARLY),
              primary_selector="strict",secondary_selector="trade",
              primary=["aligned_frozen minus old_frozen", "aligned_joint minus aligned_mse"],
              benchmark=["aligned_mse minus aligned_frozen","qp","acopf"],
              protocol="Physical candidate penalty identical at t+1..4. Only applied cost shifts. In frozen modes no predictor updates; MSE mode detaches decision path; joint uses both execution and proposal feedback plus same MSE.",
              dates_select=SELECT,dates_audit=list(range(20100427,20100431)),
              budget="Each active seed runs all four models for the same full epochs. Minimum two, maximum four. Extend while either selector improves materially at any of last four quarterly checks for any mode. Record ceiling explicitly.",
              strict="4 physical endpoints <= seed start (numeric allowance 1e-10), then least mean energy. Selection dates only.",
              confirmation="All complete September dates; chronological paired bootstrap, average seeds per date; report all seeds. No tuning on September."))
    save_json(CONF/"lock.json",dict(dates=[r[0] for r in draws],archive=base.joint.sha(archive),
              source="https://midcdmz.nlr.gov/oahu_archive/rawdata/Oahu_GHI/201009.zip",
              windows=base.joint.sha(CONF/"windows.npz"),manifest=base.joint.sha(CONF/"windows.csv"),
              protocol=base.joint.sha(OUT/"protocol.json")))


def tag(mode,step):
    return f"timing_{mode}_s{step}"


def evaluate_at(actor,fore,mode,seed,step):
    name=tag(mode,step)
    full.atomic_save(dict(actor=actor.state_dict(),fore=fore.state_dict(),selected=step),OUT/f"{name}_{seed}.pt")
    rows=[]
    for date in SELECT:
        if step==0:
            file=dev.OUT/f"d{date}_{dev.canonical('A','economic',seed,0)}_{seed}.csv"
            initial=torch.load(dev.OUT/f"{dev.canonical('A','economic',seed,0)}_{seed}.pt",weights_only=True)
            assert all(torch.equal(v,initial[k][n]) for k,model in (("actor",actor),("fore",fore)) for n,v in model.state_dict().items())
        else:
            file=OUT/f"d{date}_{name}_{seed}.csv"
            if not file.exists():
                base.replay(name,seed,date,directory=OUT)
        rows.append(dev.metrics(file))
    vals=pd.DataFrame(rows).mean()
    return dict(step=step,**vals.to_dict(),trade=float(.06*vals.kwh+4*(.1*vals.voltage_s/600+
                50*vals.voltage_peak+.2*vals.overload_s/600+20*vals.overload_peak)))


def epoch_job(job):
    mode,seed,epoch=job
    folder=OUT/f"{mode}_{seed}"
    folder.mkdir(exist_ok=True)
    marker=folder/"resume.pt"
    data=base.study_data()
    actor,fore=base.networks(seed,data.model)
    learn=mode in ("aligned_mse","aligned_joint")
    fore.requires_grad_(learn)
    opt=torch.optim.Adam([{"params":actor.parameters(),"lr":.0005},{"params":fore.parameters(),"lr":.00015}])
    if marker.exists():
        old=torch.load(marker,weights_only=False)
        if old["epoch"] >= epoch: return
        assert old["epoch"]==epoch-1
        actor.load_state_dict(old["actor"]); fore.load_state_dict(old["fore"]); opt.load_state_dict(old["opt"])
        hist,step=old["history"],old["step"]
    else:
        assert epoch==1
        hist=[dict(**evaluate_at(actor,fore,mode,seed,0),quarterly=True)]
        step=0
    rng=np.random.default_rng([seed,epoch,54])
    sequence=[];carried={};quarter=1
    for ids in full.schedule(data,seed,epoch):
        terms=[]
        for i in ids:
            last=carried.pop(i,data.d["old"][i])
            signal,end,_=full.sample(data,i,last,actor,fore,rng,"C",4.,forecast_feedback=mode=="aligned_joint",
                                     applied_interval=mode!="old_frozen")
            if i>=data.starts[0] and (i-data.starts[0]+1)%120:
                carried[i+1]=end
            terms.append(signal)
        ix=rng.choice(data.train,128,replace=False)
        mse=(fore(full.short.tensor(data.d["past"][ix]))-full.short.tensor(data.d["target"][ix])).square().mean()
        opt.zero_grad()
        (torch.stack(terms).mean()+10*mse).backward()
        for net in (actor,fore): torch.nn.utils.clip_grad_norm_(net.parameters(),3.,error_if_nonfinite=True)
        opt.step(); step+=1;sequence.extend(ids)
        quarterly=4*len(sequence)>=quarter*len(data.train)
        if quarterly: quarter+=1
        if quarterly or step in dev.EARLY:
            hist.append(dict(**evaluate_at(actor,fore,mode,seed,step),quarterly=quarterly))
            pd.DataFrame(hist).to_csv(folder/"validation.csv",index=False)
    assert not carried and quarter==5
    np.testing.assert_array_equal(np.sort(sequence),np.sort(data.train))
    np.save(folder/f"order_{epoch}.npy",np.asarray(sequence,dtype=np.int32))
    full.atomic_save(dict(epoch=epoch,step=step,history=hist,actor=actor.state_dict(),fore=fore.state_dict(),
                          opt=opt.state_dict()),marker)
    return mode,seed,epoch


def picked(z,selector):
    if selector=="strict":
        z=z[(z[dev.RISK]<=z.iloc[0][dev.RISK]+1e-10).all(axis=1)]
    col="trade" if selector=="trade" else "kwh"
    idx=z[col].idxmin()
    return int(z.loc[idx,"step"]),float(z.loc[idx,col])


def train():
    active=set(full.SEEDS); stop={}
    for epoch in range(1,MAX_EPOCHS+1):
        dev.run_pool(epoch_job,[(m,s,epoch) for m,s in JOBS if s in active],len(active)*len(MODES))
        for seed in sorted(active.copy()):
            orders=[base.joint.sha(OUT/f"{m}_{seed}"/f"order_{epoch}.npy") for m in MODES]
            assert len(set(orders))==1
            if epoch<2: continue
            improved=False
            for mode in MODES:
                z=pd.read_csv(OUT/f"{mode}_{seed}"/"validation.csv")
                # On resumed runs routines may already contain later epochs; use this epoch only.
                q=z.index[z.quarterly]
                start,end=q[4*(epoch-1)],q[4*epoch]
                for selector in ("trade","strict"):
                    _,prev=picked(z.iloc[:start+1],selector)
                    _,now=picked(z.iloc[:end+1],selector)
                    improved |= now < prev-1e-6
            if not improved or epoch==MAX_EPOCHS:
                stop[seed]=dict(epochs=epoch,reason="budget ceiling" if improved else "shared plateau")
                active.remove(seed)
        if not active: break
    save_json(OUT/"stopping.json",stop)
    selections=[]
    for mode,seed in JOBS:
        z=pd.read_csv(OUT/f"{mode}_{seed}"/"validation.csv")
        for selector in ("strict","trade"):
            step,_=picked(z,selector)
            policy=tag(mode,step)
            if step==0: policy=tag("old_frozen",0)
            filename=f"{policy}_{seed}.pt"
            shutil.copyfile(OUT/filename,CONF/filename)
            selections.append(dict(mode=mode,seed=seed,selector=selector,step=step,policy=policy,
                                   sha=base.joint.sha(CONF/filename)))
    pd.DataFrame(selections).to_csv(OUT/"locked_selection.csv",index=False)


def test_job(job):
    directory,policy,seed,date=job
    path=directory/f"d{date}_{policy}_{seed}.csv"
    if not path.exists(): base.replay(policy,seed,date,directory=directory,loss_components=True)
    return dev.metrics(path)


def confirm():
    selection=pd.read_csv(OUT/"locked_selection.csv")
    workload=[]
    for mode,seed in JOBS:
        folder=OUT/f"{mode}_{seed}"
        orders=sorted(folder.glob("order_*.npy"))
        exposure=sum(len(np.load(path)) for path in orders)
        hist=pd.read_csv(folder/"validation.csv")
        # Inclusive wall time includes validation and synchronization between full passes.
        elapsed=(folder/"resume.pt").stat().st_mtime-(OUT/f"{tag(mode,0)}_{seed}.pt").stat().st_mtime
        workload.append(dict(mode=mode,seed=seed,full_passes=len(orders),training_states=exposure,
            training_steps=int(hist.step.max()),candidate_actions_scored=exposure*(1+2*full.short.prior.DIRECTIONS),
            elapsed_including_validation_and_barriers_s=elapsed))
    pd.DataFrame(workload).to_csv(OUT/"workload.csv",index=False)
    policies=set(zip(selection.policy,selection.seed))|{("qp",0),("acopf",0)}
    dates=json.loads((CONF/"lock.json").read_text())["dates"]
    assert all(base.joint.sha(CONF/f"{r.policy}_{r.seed}.pt")==r.sha for r in selection.itertuples())
    jobs=[(directory,p,s,d) for directory,days in [(OUT,range(20100427,20100431)),(CONF,dates)]
          for p,s in sorted(policies) for d in days]
    dev.run_pool(test_job,jobs,24)
    rows=[]
    for directory,days in [(OUT,range(20100427,20100431)),(CONF,dates)]:
        for day in days:
            for r in selection.itertuples():
                rows.append(dict(date=day,mode=r.mode,seed=r.seed,selector=r.selector,step=r.step,
                                 **dev.metrics(directory/f"d{day}_{r.policy}_{r.seed}.csv")))
            for p in ("qp","acopf"):
                rows.append(dict(date=day,mode=p,seed=0,selector="fixed",step=-1,
                                 **dev.metrics(directory/f"d{day}_{p}_0.csv")))
    d=pd.DataFrame(rows)
    d.to_csv(OUT/"by_date_seed.csv",index=False)
    s=d[d.date>=20100901]
    by=s.groupby(["mode","selector","date"])[ENDPOINTS].mean()
    score=by.groupby(level=[0,1]).mean()
    score.to_csv(OUT/"september_summary.csv")
    contrasts=[]; ix=np.random.default_rng(211124).integers(len(dates),size=(10000,len(dates)))
    block_start=np.random.default_rng(211125).integers(len(dates),size=(10000,(len(dates)+2)//3))
    blocks=((block_start[:,:,None]+np.arange(3))%len(dates)).reshape(10000,-1)[:,:len(dates)]
    for sel in ("strict","trade"):
        for a,b in [("aligned_frozen","old_frozen"),("aligned_joint","aligned_mse"),
                    ("aligned_mse","aligned_frozen"),("aligned_joint","qp"),("aligned_joint","acopf")]:
            refsel="fixed" if b in ("qp","acopf") else sel
            left=by.xs((a,sel),level=("mode","selector")).sort_index()
            right=by.xs((b,refsel),level=("mode","selector")).sort_index()
            assert list(left.index)==list(right.index)==sorted(dates)
            diff=left-right
            for col in ENDPOINTS:
                vals=diff[col].to_numpy()
                lo,hi=np.quantile(vals[ix].mean(1),[.025,.975])
                block_lo,block_hi=np.quantile(vals[blocks].mean(1),[.025,.975])
                contrasts.append((sel,a,b,col,vals.mean(),lo,hi,block_lo,block_hi))
    paired=pd.DataFrame(contrasts,columns=["selector","method","reference","metric","delta","low","high",
                                           "block3_low","block3_high"])
    paired.to_csv(OUT/"paired.csv",index=False)
    text=["# IEEE123 生效区间与联合训练","",f"九月封存日期 {len(dates)} 个，每窗十分钟。下表为逐日先平均种子后再平均日期的结果。","",
          "|组别|选模|损耗 kWh/窗|越限秒/窗|最大电压越界均值 p.u.|过载秒/窗|","|---|---|---:|---:|---:|---:|"]
    for (mode,sel),r in score.iterrows():
        text.append(f"|{mode}|{sel}|{r.kwh:.5f}|{r.voltage_s:.2f}|{r.voltage_peak:.6f}|{r.overload_s:.2f}|")
    text+=["","## 严格选模的两项主要对照","",
           "差值为前者减后者，负数表示该项下降。区间按日期配对，相邻三天成块重采样。","",
           "|对照|总损耗差 Wh/窗，95% 区间|越限秒差|过载秒差|四项风险差区间上端均不大于零|",
           "|---|---:|---:|---:|---|"]
    for method,reference in (("aligned_frozen","old_frozen"),("aligned_joint","aligned_mse")):
        comp=paired[(paired.selector=="strict")&(paired.method==method)&
                    (paired.reference==reference)].set_index("metric")
        energy=comp.loc["kwh",["delta","block3_low","block3_high"]].astype(float)*1000
        safe=bool((comp.loc[dev.RISK,"block3_high"]<=1e-10).all())
        text.append(f"|{method} − {reference}|{energy.delta:+.2f} [{energy.block3_low:+.2f}, {energy.block3_high:+.2f}]|"
                    f"{comp.loc['voltage_s','delta']:+.2f}|{comp.loc['overload_s','delta']:+.2f}|{'是' if safe else '否'}|")
    text+=["","按日期配对区间见 paired.csv，逐种子及过载幅值见 by_date_seed.csv。训练早停或预算封顶见 stopping.json。"]
    import ieee123_interval_fixed_qp as fixed_qp
    fixed_qp.confirm()
    text+=["","同一 QP 采用历史初始预测或各自训练后的预测，按种子配对的结果见 fixed_qp/paired.csv。该补充组使用已归档的预测端训练预算。"]
    (OUT/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__=="__main__":
    {"prepare":prepare,"train":train,"confirm":confirm}[sys.argv[1]]()
