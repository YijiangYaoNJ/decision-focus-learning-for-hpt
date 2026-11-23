"""Independent date-paired test of frozen-forecast dispatch inputs on IEEE123."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import ieee123_chen_replay as engine
import ieee123_matched_inputs as study
from ieee123_chen_windows import ROOT, choose, joint, local, phase_power, predictor, ZERO
from ieee123_interval_data import MAY

OUT = study.OUT / "august_holdout"
SEEDS = (11, 23, 37)
ARMS = [(mode, seed) for seed in SEEDS for mode in study.MODES] + [("qp", 0)]
ARCHIVE = MAY.with_name("oahu_201008.zip")
REF = joint.OUT / "init_11.pt"


def fingerprints():
    paths = [REF, Path(study.__file__), Path(engine.__file__), ROOT/"tools/ieee123_chen_projection.py",
             ROOT/"configs/ieee123_service_areas.json"]
    paths.extend(study.OUT / f"{mode}_{seed}.pt" for mode, seed in ARMS if seed)
    return {str(p.relative_to(ROOT)): joint.sha(p) for p in paths}


def descriptor(series, slot):
    pq = np.stack((series[slot].real, series[slot].imag))
    total = pq.sum(axis=1)
    return total, pq / total[:, None]


def spatial(series, slot, refs):
    total, shares = descriptor(series, slot)
    matches = []
    for other in refs:
        if other // 96 == slot // 96:
            continue
        base, dist = descriptor(series, other)
        relative = np.abs(total-base) / np.maximum(np.abs(base), 1e-8)
        if relative.max() <= .05:
            matches.append((np.linalg.norm(relative), int(other), np.abs(shares-dist).sum(1).mean()/2))
    if not matches:
        return -1, np.nan
    _, other, distance = min(matches)
    return other, distance


def used_dates():
    # Restrict to experiment date fields and manifest tokens, skipping raw numeric meters.
    pattern = re.compile(r"(?<![\d.])2010-?08-?(0[1-9]|[12]\d|3[01])(?![\d.])")
    used = set()
    for folder in (ROOT/"runs").glob("ieee123*"):
        if folder == study.OUT:
            continue
        for p in folder.rglob("*"):
            used.update(pattern.findall(p.name))
            if p.is_file() and p.suffix in {".csv", ".json", ".md"} and p.stat().st_size < 16000000:
                used.update(pattern.findall(p.read_text(encoding="utf-8-sig",errors="replace")))
    return sorted(int("201008"+d) for d in used)


def prepare():
    OUT.mkdir(exist_ok=True)
    manifest = OUT/"lock.json"
    if manifest.exists():
        verify()
        return
    data = study.prior.April()
    used = used_dates()
    draws = [r for r in choose(ARCHIVE,1,np.random.default_rng(82311)) if r[0] not in used]
    if len(draws) < 12:
        raise ValueError(f"Only {len(draws)} dates remain after screening {used}")
    rng = np.random.default_rng(82312)
    selected = [draws[int(rng.choice(block))] for block in np.array_split(np.arange(len(draws)),12)]
    forecasts, init_hash = predictor(np.stack([r[3] for r in selected]))
    series = data.model.series
    refs = np.unique(data.d["slot"][data.train])
    dev = []
    for split, slots in [("train",refs),("validation",np.unique(data.d["slot"][data.val]))]:
        for slot in slots:
            other, distance = spatial(series,slot,refs)
            dev.append((split,slot,other,distance))
    dev = pd.DataFrame(dev,columns=["split","slot","reference","share_distance"])
    dev.to_csv(OUT/"development_spatial.csv",index=False)
    distances = dev.loc[dev.split=="train","share_distance"].dropna()
    threshold = max(.01,float(distances.quantile(.75))) if len(distances) else .01
    rows, arrays = [], {"fixed": data.fixed}
    for (date, start, slot, _, seq), forecast in zip(selected,forecasts):
        state = data.model.step(slot,ZERO,"high",generation_kw=phase_power(forecast))
        arrays.update({f"{date}_pv":seq, f"{date}_taps":state["taps"], f"{date}_initial":local(state,data.model,ZERO)})
        other, distance = spatial(series,slot,refs)
        band = "unmatched" if not np.isfinite(distance) else ("high" if distance>=threshold else "ordinary")
        rows.append((date,start,slot,other,distance,band))
    pd.DataFrame(rows,columns=["date","second","slot","april_reference","share_distance","stratum"]).to_csv(OUT/"windows.csv",index=False)
    np.savez_compressed(OUT/"windows.npz",**arrays)
    rules = dict(archive=joint.sha(ARCHIVE), source="https://midcdmz.nlr.gov/oahu_archive/rawdata/Oahu_GHI/201008.zip",
                 previous_august_dates=used,eligible=[r[0] for r in draws],seeds=SEEDS,date_seeds=[82311,82312],
                 initial_forecasts=init_hash,files=fingerprints(),windows=joint.sha(OUT/"windows.npz"),
                 manifest=joint.sha(OUT/"windows.csv"),share_threshold=threshold,
                 stratum="closest other-day April training slot within 5% on both P and Q; mean P/Q total variation >= max(1%, April training Q75)",
                 primary="B-A; B-QP and A-QP are additional. Average seeds within date; bootstrap dates 10000 times, seed 82501.",
                 safety="No increase in violation seconds, max voltage gap, overload seconds or max overload gap; upper 95% paired interval <=0 on each; kWh upper limit <0 for energy gain. Descriptive for sampled windows.",
                 scoring="600 metered seconds, 5s online clock, voltage band 0.95..1.05, overload util >1+1e-7; also retain actual peak utilization.")
    manifest.write_text(json.dumps(rules,indent=2))
    print("Locked",rows,flush=True)


def verify():
    rules = json.loads((OUT/"lock.json").read_text())
    assert fingerprints()==rules["files"]
    for name, key in [("windows.npz","windows"),("windows.csv","manifest")]:
        assert joint.sha(OUT/name)==rules[key]
    return pd.read_csv(OUT/"windows.csv")


def policies(model):
    actions = {("qp",0):lambda *args: study.DecisionInput(*args).qp()}
    for mode, seed in ARMS[:-1]:
        _, base = joint.load_models(seed,joint.OUT/f"init_{seed}.pt")
        net = study.MatchedActor(seed,base.state_dict(),len(model.lp))
        net.load_state_dict(torch.load(study.OUT/f"{mode}_{seed}.pt",weights_only=True))
        net.eval()
        def propose(*args, net=net, mode=mode):
            return study.prior.disk(net(*study.DecisionInput(*args).network(),mode=mode).numpy())[0]
        actions[mode,seed] = propose
    return actions


def replay():
    windows = verify()
    _, model, *_ = engine.build()
    ref = engine.forecast_net(REF)
    actions = policies(model)
    tag = joint.sha(OUT/"lock.json")
    with np.load(OUT/"windows.npz") as z:
        for j, row in enumerate(windows.itertuples()):
            file = OUT/f"replay_{row.date}.npz"
            if file.exists():
                with np.load(file) as previous:
                    assert str(previous["tag"])==tag
                continue
            pv = z[f"{row.date}_pv"]
            truth = phase_power(pv[10:610])
            plant = engine.BatchAC(model,int(row.slot),tuple(z[f"{row.date}_taps"]))
            initial, fixed = z[f"{row.date}_initial"], z["fixed"]
            old = {k:initial.copy() for k in ARMS}
            last_voltage = {k:None for k in ARMS}
            metrics = np.zeros((7,600,6))
            commands = np.zeros((7,600,54))
            timing = np.zeros((7,120,3))
            for k in ARMS:
                engine.issue(None,plant,initial,fixed,np.tile(pv[0],(11,1)),ref,ref,actions[k])
            for cycle in range(120):
                offset = 5*cycle
                cache = {}
                for arm in np.roll(np.arange(7),-(j+cycle)%7):
                    k = ARMS[arm]
                    past = pv[offset:offset+11]
                    events, _, _, elapsed, propose_time, failure = engine.issue(None,plant,old[k],fixed,past,ref,ref,actions[k])
                    timing[arm,cycle] = elapsed, propose_time, failure
                    for s in range(5):
                        available = [a for stamp, _, a in events if stamp<=s]
                        cmd = available[-1] if available else old[k]
                        key = s, cmd.tobytes()
                        if key not in cache:
                            try:
                                state=model.solve(int(row.slot),cmd,"high",plant.taps,generation_kw=truth[offset+s],tolerance=1e-8,max_iterations=250)
                            except ArithmeticError:
                                state=model.solve(int(row.slot),cmd,"high",plant.taps,generation_kw=truth[offset+s],start=last_voltage[k],tolerance=1e-8,max_iterations=1500)
                            m=engine.metrics(model,state,int(row.slot),"high","holdout",model.ratings.ravel())
                            cache[key] = state["v"],[m[f] for f in engine.FIELDS]
                        last_voltage[k],metrics[arm,offset+s] = cache[key]
                        commands[arm,offset+s] = cmd
                    if events:
                        old[k] = events[-1][2]
            assert np.isfinite(metrics).all() and np.isfinite(timing[:,:,0]).all()
            tmp=file.with_suffix(".part.npz")
            np.savez_compressed(tmp,tag=tag,meters=metrics,actions=commands,timing=timing)
            tmp.replace(file)
            print("Completed",row.date,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","replay","report"])
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT/"src"))
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.stage=="prepare":
            prepare()
        elif args.stage=="replay":
            replay()
        else:
            from ieee123_matched_report import report
            report()
