"""April closed-loop selection study, using the original trainer and native timed replay."""
import concurrent.futures
import itertools
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ieee123_full_coverage as full
import ieee123_joint5_trajectories as trajectories
import ieee123_proposal_run as runner
from ieee123_direct_confirm_data import previous

base = full.base
OUT = full.OUT.with_name("ieee123_april_closed_validation_v1")
EARLY = (0, 1, 5, 10, 25, 50, 100)
JOBS = list(itertools.product("ABC", full.WEIGHTS, full.SEEDS))
RISK = ["voltage_s", "voltage_peak", "overload_s", "overload_peak"]


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / "lock.json").exists():
        return
    sys.path.insert(0, str(previous.ROOT / "src"))
    dates = list(range(20100423, 20100431))
    rows = trajectories.choose(trajectories.ARCHIVE, 1, np.random.default_rng(211121),
                               date_range=(dates[0], dates[-1]))
    assert [r[0] for r in rows] == dates
    prediction, _ = previous.predictor(np.stack([r[3] for r in rows]))
    model = base.plant_model()
    data, manifest = {}, []
    for (date, second, slot, _, pv), fore in zip(rows, prediction):
        assert pv.shape == (611, 6)
        initial = model.step(slot, previous.ZERO, "high", generation_kw=previous.phase_power(fore))
        data[f"{date}_pv"] = pv
        data[f"{date}_initial"] = previous.local(initial, model, previous.ZERO)
        data[f"{date}_taps"] = initial["taps"]
        manifest.append((date, second, slot, "select" if date < dates[4] else "audit"))
    pd.DataFrame(manifest, columns=["date", "second", "slot", "split"]).to_csv(OUT/"windows.csv", index=False)
    np.savez_compressed(OUT/"windows.npz", **data)
    lock = dict(dates=dates, early=EARLY, seeds=full.SEEDS, weights=full.WEIGHTS,
                windows=base.joint.sha(OUT/"windows.npz"), manifest=base.joint.sha(OUT/"windows.csv"),
                archive=base.joint.sha(trajectories.ARCHIVE),
                training="Replay original full passes and stopping. Record early and quarterly weights; verify against previous validation traces.",
                primary="High-safety C vs B; report all arms, weights, seeds. April 23-26 select, 27-30 audit. Development only.",
                strict="Mean four risk endpoints <= paired seed initial (+1e-10 arithmetic tolerance); minimize energy among feasible.",
                trade="Mean kW/100 + w*(0.1*voltage_seconds/600 + 50*window_voltage_peak + 0.2*overload_seconds/600 + 20*window_overload_peak).",
                cadence="Compare quarterly-only vs dense with identical scoring, and archived 36-state selection vs closed-loop selection. Identical step ties prefer earliest.")
    (OUT/"lock.json").write_text(json.dumps(lock, indent=2))


def policy(arm, pref, step):
    return f"dev_{arm}_{pref}_s{step}"


def fit(job):
    arm, pref, seed = job
    tag = f"{arm}_{pref}_{seed}"
    def observe(step, net, fore, quarterly):
        if step in EARLY or quarterly:
            full.atomic_save(dict(actor=net.state_dict(), fore=fore.state_dict(), selected=step),
                             OUT/f"{policy(arm,pref,step)}_{seed}.pt")
    full.train(*job, directory=OUT, observer=observe)
    original = pd.read_csv(full.OUT/tag/"validation.csv")
    replica = pd.read_csv(OUT/tag/"validation.csv")
    np.testing.assert_allclose(original, replica, atol=1e-8, rtol=1e-7, equal_nan=True)
    for order in (full.OUT/tag).glob("order_*.npy"):
        assert base.joint.sha(order) == base.joint.sha(OUT/tag/order.name)
    print("Training trajectory verified", tag, flush=True)


def metrics(path):
    d = pd.read_csv(path)
    np.testing.assert_array_equal(d.second, np.arange(0,600,5))
    assert np.isfinite(d[["ms", "kwh", *RISK]]).all().all()
    components={}
    if "network_kwh" in d:
        np.testing.assert_allclose(d.network_kwh+d.converter_kwh,d.kwh,rtol=1e-9,atol=1e-12)
        components=dict(network_kwh=d.network_kwh.sum(),converter_kwh=d.converter_kwh.sum())
    return dict(kwh=d.kwh.sum(), voltage_s=d.voltage_s.sum(), voltage_peak=d.voltage_peak.max(),
                overload_s=d.overload_s.sum(), overload_peak=d.overload_peak.max(),
                issued=d.status.eq("issued").mean(),timeouts=d.status.eq("timeout").sum(),
                p50_ms=d.ms.median(),p95_ms=d.ms.quantile(.95),**components)


def replay(task):
    name, seed, date = task
    file = OUT/f"d{date}_{name}_{seed}.csv"
    if not file.exists():
        base.replay(name, seed, date, directory=OUT)
    return metrics(file)


def canonical(arm, pref, seed, step):
    # Step-zero networks are initialized by seed only; verify identical tensors before aliasing.
    if step == 0:
        return policy("A", "economic", 0)
    return policy(arm, pref, step)


def run_pool(fn, tasks, size):
    ctx = multiprocessing.get_context("fork")
    cpus = runner.cores(size)
    queue = ctx.Queue()
    for cpu in cpus:
        queue.put(cpu)
    with concurrent.futures.ProcessPoolExecutor(len(cpus), mp_context=ctx, initializer=runner.initialize,
                                                initargs=(queue,)) as pool:
        futures = [pool.submit(fn, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def design():
    dates = json.loads((OUT/"lock.json").read_text())["dates"]
    for arm, pref, seed in JOBS:
        tag = f"{arm}_{pref}_{seed}"
        quarter = set(pd.read_csv(full.OUT/tag/"validation.csv").step.astype(int))
        for step in sorted(quarter.union(EARLY)):
            name = canonical(arm, pref, seed, step)
            for date in dates:
                yield arm, pref, seed, step, step in quarter, name, date


def evaluate():
    import time
    # Evaluate completed snapshots while long trajectories are still training.
    tasks = {(name, seed, date) for _,_,seed,_,_,name,date in design()}
    while tasks:
        ready = [t for t in sorted(tasks) if (OUT/f"{t[0]}_{t[1]}.pt").exists()]
        if ready:
            missing = [t for t in ready if not (OUT/f"d{t[2]}_{t[0]}_{t[1]}.csv").exists()]
            if missing:
                run_pool(replay, missing, min(24, len(missing)))
            tasks.difference_update(ready)
        else:
            if (OUT/"training_failed.txt").exists():
                raise RuntimeError((OUT/"training_failed.txt").read_text())
            time.sleep(15)


def train_all():
    import traceback
    try:
        run_pool(fit, JOBS, len(JOBS))
    except Exception:
        (OUT/"training_failed.txt").write_text(traceback.format_exc())
        raise
    (OUT/"training_verified.json").write_text(json.dumps({"verified":len(JOBS)}))


def report():
    assert json.loads((OUT/"training_verified.json").read_text())["verified"]==len(JOBS)
    rows = []
    for arm,pref,seed,step,quarter,name,date in design():
        if step==0 and date==20100423:
            actual=torch.load(OUT/f"{policy(arm,pref,step)}_{seed}.pt",weights_only=True)
            shared=torch.load(OUT/f"{name}_{seed}.pt",weights_only=True)
            assert all(torch.equal(v,shared[k][n]) for k in ("actor","fore") for n,v in actual[k].items())
        rows.append(dict(arm=arm,pref=pref,seed=seed,step=step,quarter=quarter,date=date,
                         **metrics(OUT/f"d{date}_{name}_{seed}.csv")))
    d=pd.DataFrame(rows)
    d["trade"]=.06*d.kwh + d.pref.map(full.WEIGHTS)*(
        .1*d.voltage_s/600+50*d.voltage_peak+.2*d.overload_s/600+20*d.overload_peak)
    d.to_csv(OUT/"by_date.csv",index=False)
    choices=[]
    targets=["kwh",*RISK,"issued","trade"]
    for (arm,pref,seed),g in d.groupby(["arm","pref","seed"]):
        sel=g[g.date<20100427].groupby("step")[targets].mean()
        audit=g[g.date>=20100427].groupby("step")[targets].mean()
        old=json.loads((full.OUT/f"{arm}_{pref}_{seed}"/"complete.json").read_text())["best_steps"]
        for selector in ("strict","trade"):
            options={"isolated":int(old[selector])}
            for cadence,steps in [("quarterly",g.loc[g.quarter,"step"].unique()),("dense",sel.index)]:
                z=sel.loc[sorted(steps)]
                if selector=="strict":
                    z=z[(z[RISK]<=sel.loc[0,RISK]+1e-10).all(axis=1)]
                options[cadence]=int(z["kwh" if selector=="strict" else "trade"].idxmin())
            for rule,step in dict(initial=0,**options).items():
                changes=(audit.loc[step]-audit.loc[0]).add_prefix("delta_").to_dict()
                choices.append(dict(arm=arm,pref=pref,seed=seed,selector=selector,rule=rule,
                                    selected=step,**audit.loc[step].to_dict(),**changes))
    table=pd.DataFrame(choices)
    table.to_csv(OUT/"selection_audit.csv",index=False)
    txt=["# IEEE123 四月开发期连续验证", "", "4 月 23–26 日选模，27–30 日核验。各日期连续 600 秒。差值均相对同种子初始化。","",
         "|选模|验证设计|选中更新模型的组数/27|四日核验平均损耗差 kWh/窗|电压越限秒差|过载秒差|",
         "|---|---|---:|---:|---:|---:|"]
    for (selector,rule),z in table.groupby(["selector","rule"]):
        txt.append(f"|{selector}|{rule}|{z.selected.ne(0).sum()}|{z.delta_kwh.mean():+.5f}|{z.delta_voltage_s.mean():+.2f}|{z.delta_overload_s.mean():+.2f}|")
    primary=table[(table.pref=="safety")&(table.rule=="dense")].groupby(["selector","arm"])[targets].mean()
    txt+=["","## 安全权重下，预测器接收额外安全反馈的增量",
          "","C 将提案安全损失同时传至动作网络和预测器，B 仅将该项传至动作网络，两者均保留执行代价反馈。",
          "","|选模|C 减 B 的损耗 Wh/窗|越限秒差|平均最大越界差 p.u.|过载秒差|",
          "|---|---:|---:|---:|---:|"]
    for sel in ("strict","trade"):
        diff=primary.loc[(sel,"C")]-primary.loc[(sel,"B")]
        txt.append(f"|{sel}|{diff.kwh*1000:+.2f}|{diff.voltage_s:+.2f}|{diff.voltage_peak:+.6f}|{diff.overload_s:+.2f}|")
    txt+=["","各组检查点、幅值、放行率和逐日期明细见 selection_audit.csv 与 by_date.csv。各组平均含共享初始化和重复选中模型，不代表独立重复次数。本实验用于开发选模。"]
    (OUT/"REPORT.md").write_text("\n".join(txt)+"\n",encoding="utf8")


if __name__=="__main__":
    {"prepare":prepare,"fit":train_all,"replay":evaluate,"report":report}[sys.argv[1]]()
