"""Lock all remaining untouched August dates before running the weight comparison."""
import json
import sys
import numpy as np
import pandas as pd
import ieee123_direct_confirm_data as base

OUT=base.OUT.with_name("ieee123_proposal_weights_v1")
WEIGHTS={"economic":0.25,"balanced":1.0,"safety":4.0}
SEEDS=(11,23,37)

def prepare():
    old=json.loads((base.OUT/"lock.json").read_text())
    used=set(old["dates"]+old["excluded_august"])
    prior=base.previous
    for file in prior.ROOT.joinpath("runs").glob("ieee123*/**/*.csv"):
        if OUT in file.parents or file.stat().st_size>64_000_000:
            continue
        if "date" in pd.read_csv(file,nrows=0).columns:
            dates=pd.read_csv(file,usecols=["date"]).date.astype(str).str.replace("-","",regex=False)
            used.update(int(d) for d in dates.unique() if d.isdigit() and len(d)==8 and d.startswith("201008"))
    OUT.mkdir(exist_ok=True)
    if (OUT/"lock.json").exists():
        raise FileExistsError("already locked")
    sys.path.insert(0,str(prior.ROOT/"src"))
    draws=[r for r in prior.choose(prior.ARCHIVE,1,np.random.default_rng(112101)) if r[0] not in used]
    assert len(draws)>0
    paths,_=prior.predictor(np.stack([r[3] for r in draws]))
    model=base.plant_model()
    arrays={}; rows=[]
    for (date,t,slot,_,pv),forecast in zip(draws,paths):
        state=model.step(slot,prior.ZERO,"high",generation_kw=prior.phase_power(forecast))
        arrays.update({f"{date}_pv":pv,f"{date}_taps":state["taps"],f"{date}_initial":prior.local(state,model,prior.ZERO)})
        rows.append((date,t,slot))
    np.savez_compressed(OUT/"windows.npz",**arrays)
    pd.DataFrame(rows,columns=["date","second","slot"]).to_csv(OUT/"windows.csv",index=False)
    lock=dict(dates=[r[0] for r in rows],excluded=sorted(used),weights=WEIGHTS,seeds=SEEDS,
        windows=prior.joint.sha(OUT/"windows.npz"),manifest=prior.joint.sha(OUT/"windows.csv"),
        train_rule="Matched 40 updates, two states per step, same samples/probes; historical auto-stop budget. Evaluate April every 10 steps including 0.",
        strict="Lexicographic validation risk excess relative to seed initialization, then energy.",
        trade="Minimum validation executed cost kW/100+w*(0.1*voltage frequency+50*voltage peak+0.2*overload frequency+20*overload peak).",
        proposal="w*0.1*(worst voltage excess / 0.003 + worst device utilization excess / 0.01) averaged across actual seconds 1..4; no penalties at t=0 or t+5. Scales are modeling choices, not relaxation of grid limits.",
        arms="A: executed feedback; B: additional proposal feedback to actor only; C: additional proposal feedback to actor and forecaster. Same AC sensitivity computations in every arm, MSE multiplier 10.",
        tests="Report both selectors, all weights. Within-date seed mean; pair by date. No test tuning. Classical methods use prior common forecast and six-frame gate.")
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2))
    print("Locked dates",lock["dates"])

if __name__=="__main__":
    prepare()
