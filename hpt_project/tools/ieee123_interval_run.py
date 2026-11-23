"""Five-second rolling comparison; no training and no access to future test power."""
import os
for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[key]="1"
import hashlib
import json
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from ieee123_interval_ac import BatchAC,Snapshot
from ieee123_interval_data import OUT,PRIOR,forecasts,phase_power
from ieee123_ramp_dispatch import qp
from ieee123_service_areas import build,metrics

METHODS=("history","snapshot","scenario")


class Deadline(Exception):
    pass


def dispatch(arm,plant,old,fixed,past,tree,library,*,margin=0.,proposal=None,details=False,deadband=.02,
             future_path=None):
    start=perf_counter()
    def check():
        if perf_counter()-start>=5: raise Deadline()
    events=[]
    pv=phase_power(past[-1])
    f=Snapshot(plant.m,plant.slot,plant.taps,pv)
    Jx=J=None
    def derivative(x):
        nonlocal Jx,J
        check()
        if Jx is None or not np.array_equal(Jx,x):
            f.at(x)
            J=plant.jac(x,pv,f.last_v)
            Jx=x.copy()
        return J
    weather=np.atleast_2d(pv)
    if arm=="scenario":
        if future_path is None:
            weather=phase_power(forecasts(tree,library,past)).reshape(-1,18)
        else:
            future=np.asarray(future_path(),dtype=float)
            if future.shape!=(5,6) or not np.isfinite(future).all() or not ((future>=0)&(future<=1)).all():
                raise ValueError("Expected five bounded six-site forecasts")
            weather=phase_power(np.vstack((past[-1],future)))
    weather,weights=np.unique(weather,axis=0,return_counts=True)
    rows={}
    voltage_rows=2*(plant.m.original_n-3+3)
    best_key=None
    proposal_ok=False
    def evaluate(x):
        nonlocal best_key
        check()
        key=x.tobytes()
        if key in rows:return rows[key]
        if arm=="scenario":
            y=plant.evaluate(x,weather)
            value=np.r_[np.average(y[:,0],weights=weights),y[:,1:].min(0)]
        else:
            value=f.at(x)
        physical_ok=min(value[1:])>=0
        if margin:
            value=value.copy()
            value[1:1+voltage_rows]-=margin
        score=(max(0.,-min(value[1:1+voltage_rows])),value[0])
        if physical_ok and (best_key is None or score[0]<best_key[0]-1e-9 or
                (abs(score[0]-best_key[0])<=1e-9 and score[1]<best_key[1]-1e-7)):
            # A cold single-case acceptance check independent of the batch model.
            actual,_=f.evaluate(x)
            if min(actual[1:])>=0:
                events.append((perf_counter()-start,x.copy(),value[0]))
                best_key=score
        rows[key]=value
        return value
    status="finished"
    try:
        y=evaluate(old)
        old_certified=bool(events)
        alt=evaluate(fixed)
        x=events[-1][1] if events else (old if min(y[1:])>=min(alt[1:]) else fixed)
        if arm=="snapshot":
            y=evaluate(x)
            if min(y[1:])<0:
                trial,_=qp(x,y,derivative(x),1,restore=True)
                if trial is not None and min(evaluate(trial)[1:])>min(y[1:]):x=trial
            res=minimize(lambda a:evaluate(a)[0],x,jac=lambda a:derivative(a)[0],
                method="SLSQP",bounds=[(-1,1)]*54,
                constraints=[dict(type="ineq",fun=lambda a:evaluate(a)[1:]-2e-5,jac=lambda a:derivative(a)[1:])],
                options=dict(maxiter=45,ftol=1e-4))
            evaluate(res.x)
            status=str(res.message)
        elif arm in ("economic_qp","sensitivity","scenario"):
            # Common-Jacobian successive QP; each line-search point is rescored
            # on every distinct future using nonlinear AC (including current PV).
            for _ in range(8 if arm=="scenario" else 1):
                y=evaluate(x)
                gradient=derivative(x)
                if arm=="sensitivity":
                    gradient=gradient.copy()
                    gradient[0]=10*x  # minimum absolute effort under linearized limits
                trial,status=qp(x,y,gradient,1,restore=arm!="sensitivity" and min(y[1:])<0)
                if trial is None or np.max(abs(trial-x))<2e-4:break
                merit=lambda z:z[0]+1e6*max(0,2e-5-min(z[1:]))
                for alpha in (1,.5,.25,.125,.0625):
                    proposal=x+alpha*(trial-x)
                    if merit(evaluate(proposal))<merit(y)-1e-5:
                        x=proposal
                        break
                else:break
        elif arm=="droop" or proposal is not None:
            if arm=="droop":
                _,s=f.evaluate(old)
                v=s["v"][plant.m.secondary].copy()
                v[0]-=v[0].mean()
                err=1-abs(v)/plant.m.bases[plant.m.secondary]/(1+.1*old.reshape(6,3,3)[:,:,0])
                step=np.sign(err)*np.maximum(abs(err)-deadband,0.)
                a=np.zeros((6,3,3))
                a[:,:,0]=np.clip(6*step,-1,1)
                a[:,:,2]=np.clip(2*step,-.25,.25)
                candidate=a.ravel()
            else:
                candidate=proposal()
            value=evaluate(candidate)
            proposal_ok=(min(value[1:1+voltage_rows])>=-margin and min(value[1+voltage_rows:])>=0)
    except (Deadline,ArithmeticError) as err:
        status=type(err).__name__
    elapsed=perf_counter()-start
    due=[e for e in events if e[0]<=5]
    new=next((t for t,a,_ in due if np.max(abs(a-old))>1e-8),np.nan)
    times=dict(elapsed_s=elapsed,first_s=due[0][0] if due else np.nan,
        first_new_s=new,old_certified=bool(due and np.array_equal(due[0][1],old)),
        status=status,unique_weather=len(weather))
    if details:
        times["proposal_ok"]=proposal_ok
        times["proposal_selected"]=bool((arm=="droop" or proposal is not None) and due and
                                        "candidate" in locals() and np.array_equal(candidate,due[-1][1]))
    return due,times


def run():
    lock=json.loads((OUT/"lock.json").read_text())
    for filename,key in [("windows.csv","windows_sha"),("windows.npz","fixed_inputs_sha"),("development.npz","dev_sha")]:
        assert hashlib.sha256((OUT/filename).read_bytes()).hexdigest()==lock[key]
    w=pd.read_csv(OUT/"windows.csv",dtype={"date":str})
    data=np.load(OUT/"windows.npz")
    dev=np.load(OUT/"development.npz")
    tree=cKDTree(dev["features"])
    library=dev["futures"]
    fixed=data["fixed"]
    _,m,*_=build()
    warm=False
    for i,row in enumerate(w.itertuples()):
        d=row.date
        folder=OUT/d
        folder.mkdir(exist_ok=True)
        if (folder/"complete.json").exists():continue
        taps=tuple(data[d+"_taps"])
        plant=BatchAC(m,row.slot,taps)
        initial=data[d+"_initial"]
        seq=data[d+"_pv"]
        previous={arm:initial.copy() for arm in METHODS}
        if not warm:
            # Warm the implementation on a constant pre-window measurement.
            pre=np.tile(seq[0],(11,1))
            for arm in METHODS: dispatch(arm,plant,initial,fixed,pre,tree,library)
            warm=True
        grid=phase_power(seq)
        anchor=Snapshot(m,row.slot,taps,grid[0])
        commands=np.vstack([initial,fixed,.5*(initial+fixed)])
        reference=np.stack([anchor.evaluate(x)[0] for x in commands])
        np.testing.assert_allclose(plant.evaluate(commands,grid[0]),reference,rtol=1e-6,atol=2e-5)
        test_weather=grid[[0,1,7]]
        expected=np.stack([Snapshot(m,row.slot,taps,g).evaluate(initial)[0] for g in test_weather])
        np.testing.assert_allclose(plant.evaluate(initial,test_weather),expected,rtol=1e-6,atol=2e-5)
        cycles=[]; seconds=[]; actionbank=[]; eventlog=[]
        for offset in range(0,600,5):
            past=seq[offset:offset+PRIOR+1]
            for j in range(3):
                arm=METHODS[(j+i+offset//5)%3]
                old=previous[arm]
                due,times=dispatch(arm,plant,old,fixed,past,tree,library)
                common=dict(date=d,offset=offset,method=arm)
                cycles.append(dict(**common,**times))
                for t,x,loss in due:
                    eventlog.append(dict(**common,t=t,id=len(actionbank),forecast_loss_kw=loss))
                    actionbank.append(x)
                for sec in range(5):
                    ready=[evt for evt in due if evt[0]<=sec]
                    a=ready[-1][1] if ready else old
                    state=m.solve(row.slot,a,"high",taps,generation_kw=grid[offset+PRIOR+sec],
                                  tolerance=1e-8,max_iterations=250)
                    obs=metrics(m,state,row.slot,"high",arm,m.ratings.ravel())
                    seconds.append(dict(**common,s=sec,**{k:obs[k] for k in
                        ("vmin","vmax","excursion","violation","overload","main_util","port_util","loss_kw","network_kw","heat_kw")}))
                previous[arm]=due[-1][1] if due else old
            if offset%100==95:print(d,offset+5,flush=True)
        pd.DataFrame(cycles).to_csv(folder/"cycles.csv",index=False)
        pd.DataFrame(seconds).to_csv(folder/"seconds.csv",index=False)
        pd.DataFrame(eventlog).to_csv(folder/"events.csv",index=False)
        np.savez_compressed(folder/"actions.npz",actions=np.asarray(actionbank))
        (folder/"complete.json").write_text(json.dumps(dict(cycles=len(cycles),seconds=len(seconds),
            actions=len(actionbank),snapshot_batch_parity=True)))


if __name__=="__main__":
    with threadpool_limits(1):run()
