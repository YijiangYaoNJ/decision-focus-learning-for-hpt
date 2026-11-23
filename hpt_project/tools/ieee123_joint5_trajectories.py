"""Expand the April dispatch pool and replay paired controllers on blind June windows."""
from datetime import datetime
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import ieee123_joint5 as joint
from ieee123_interval_data import phase_power
from ieee123_interval_run import dispatch
from ieee123_preference_replay import features, policy, run
from ieee123_ramp_windows import ROOT, ARCHIVE, SIX, day_data
from ieee123_service_areas import build, local, ZERO

OUT = ROOT / "runs/ieee123_joint5_trajectories_v1"
JUNE = ARCHIVE.with_name("oahu_201006.zip")
GROUPS = ("original", "expanded")


def choose(archive_path, per_day, rng, excluded=(), *, date_range=None):
    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            date = int(name[:8])
            if date_range is not None:
                if not date_range[0] <= date <= date_range[1]:
                    continue
            elif archive_path == ARCHIVE and date > 20100422:
                continue
            p = day_data(archive, name)
            candidates = []
            for boundary in range(36000, 50400, 900):
                t = boundary + 120
                if (date, t) in excluded:
                    continue
                h = p[boundary-3600:boundary].reshape(4, 900, 9)
                seq = p[t-10:t+601, SIX]
                if np.isfinite(h).all() and np.isfinite(seq).all():
                    slot = (datetime.strptime(str(date), "%Y%m%d").timetuple().tm_yday-1)*96 + t//900
                    angle = 2*np.pi*slot/96
                    candidates.append((date, t, slot, np.r_[h.mean(1)[::-1].ravel(), np.sin(angle), np.cos(angle)], seq))
            for i in rng.choice(len(candidates), min(per_day, len(candidates)), replace=False):
                rows.append(candidates[i])
    return rows


def prepare():
    from ieee123_fast_ramps import predictor
    old = dict(np.load(joint.PREVIOUS / "april_examples.npz"))
    original = dict(np.load(joint.OUT / "april.npz"))
    locked = json.loads((joint.SOURCE / "lock.json").read_text())
    assert joint.sha(ARCHIVE) == locked["april_sha"]
    _, base = joint.fixed_data()
    fixed = base["fixed"]
    train = choose(ARCHIVE, 2, np.random.default_rng(2911), set(zip(old["date"], old["second"])))
    eligible = choose(JUNE, 1, np.random.default_rng(2912))
    # Sample one available date per chronological block across the whole month.
    rng = np.random.default_rng(2913)
    test = [eligible[rng.choice(block)] for block in np.array_split(np.arange(len(eligible)), 10)]
    groups = train + test
    slow, hashes = predictor(np.stack([r[3] for r in groups]))
    assert hashes == locked["predictors"]
    manifest = pd.DataFrame([(date,t,slot,"train" if i<len(train) else "test")
                             for i,(date,t,slot,_,_) in enumerate(groups)],columns=["date","second","slot","split"])
    manifest.to_csv(OUT/"manifest.csv",index=False)
    print("Locked windows",manifest.groupby("split").size().to_dict(),flush=True)
    _, model, *_ = build()
    weights = dict(np.load(joint.PREVIOUS/"preference_11.npz"))
    assert joint.sha(joint.PREVIOUS/"preference_11.npz") == json.loads((joint.PREVIOUS/"preference_11.json").read_text())["weights_sha256"]
    arrays = {k:list(v) for k,v in old.items()}
    targets = list(original["physical_truth"])
    applied = []
    test_data = {"fixed":fixed}
    for i,((date,t,slot,_,seq),forecast) in enumerate(zip(groups,slow)):
        state = model.step(slot,ZERO,"high",generation_kw=phase_power(forecast))
        taps = tuple(state["taps"])
        previous = local(state,model,ZERO)
        if i>=len(train):
            for key,v in (("pv",seq[:-1]),("initial",previous),("taps",taps)):
                test_data[f"{date}_{key}"]=v
            continue
        plant = joint.BatchAC(model,slot,taps)
        for k in range(0,600,5):
            past = seq[k:k+11]
            x = features(model,slot,taps,past)
            for key,value in dict(date=date,second=t+k,slot=slot,taps=taps,old=previous,past=past,
                                  truth=seq[k+10:k+15],feat=x).items():
                arrays[key].append(value)
            targets.append(seq[k+11:k+16])
            events,_ = dispatch("behavior",plant,previous,fixed,past,None,None,margin=.003,
                                proposal=lambda:policy(weights,x,1))
            if events: previous=events[-1][1]
            applied.append(previous)
        print("training trajectory",i+1,len(train),flush=True)
    np.savez_compressed(OUT/"cases.npz",**arrays,target=targets,applied=applied)
    np.savez_compressed(OUT/"june.npz",**test_data)
    lock = dict(april_sha=joint.sha(ARCHIVE),june_sha=joint.sha(JUNE),eligible_june_dates=len(eligible),
                predictors=hashes,original_cases=joint.sha(joint.PREVIOUS/"april_examples.npz"),
                source_url="https://midcdmz.nlr.gov/oahu_archive/rawdata/Oahu_GHI/201006.zip",
                test_selection="one random complete window per available date; one date per chronological decile",
                files={k:joint.sha(OUT/k) for k in ("cases.npz","june.npz","manifest.csv")})
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2))


def verify_file(name):
    lock=json.loads((OUT/"lock.json").read_text())
    assert joint.sha(OUT/name)==lock["files"][name]


def train():
    verify_file("cases.npz")
    d=dict(np.load(OUT/"cases.npz"))
    n=len(d["applied"])
    assert n%120==0
    np.testing.assert_array_equal(d["old"][-n:].reshape(-1,120,54)[:,1:],
                                  d["applied"].reshape(-1,120,54)[:,:-1])
    valid=np.isfinite(d["target"]).all(axis=(1,2))
    tr=np.flatnonzero(valid & (d["date"]<=20100422))
    va=np.flatnonzero(valid & (d["date"]>20100422))
    assert len(tr)==94+n and len(va)==36 and len(set(zip(d["date"][tr],d["second"][tr])))==len(tr)
    _,model,*_=build()
    z=dict(np.load(joint.OUT/"april.npz"))
    cases=joint.TrainingCases(model,d,d["target"],np.flatnonzero(valid))
    for seed in joint.SEEDS:
        for arm in joint.ARMS:
            joint.train(seed,arm,z,d,cases,tr,va,output=OUT,step_limit=50,fixed_steps=True)


def replay(group):
    verify_file("manifest.csv")
    verify_file("june.npz")
    ws=pd.read_csv(OUT/"manifest.csv",dtype={"date":str}).query("split=='test'")
    def load(name,seed):
        arm=name.split("_",1)[1]
        return joint.loader(arm,seed,joint.OUT if group=="original" else OUT)
    run([f"{group}_{arm}" for arm in joint.ARMS],policy_loader=load,output=OUT,
        case_data=lambda:(ws,np.load(OUT/"june.npz")))


def main(stage):
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/"src"))
    with threadpool_limits(1):
        if stage=="prepare":prepare()
        elif stage=="train":train()
        elif stage in GROUPS:replay(stage)
        elif stage=="report":
            from ieee123_trajectory_report import report
            report()
        else:raise ValueError(stage)


if __name__=="__main__":
    main(sys.argv[1])
