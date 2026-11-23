"""Run the locked study; identical deployed tensors share their timed replay."""
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import ieee123_direct_confirm as base
from ieee123_direct_confirm_run import cores
from ieee123_proposal_train import train
from ieee123_proposal_windows import OUT, WEIGHTS, SEEDS


def fit(job):
    arm,pref,seed=job
    if not (OUT/f"{arm}_{pref}_{seed}_updates.csv").exists():
        train(arm,pref,seed)
    return job


def aliases():
    rows=[];mapping={}
    for arm,pref,selector,seed in itertools.product("ABC",WEIGHTS,("strict","trade"),SEEDS):
        policy=f"{arm}_{pref}_{selector}"
        state=torch.load(OUT/f"{policy}_{seed}.pt",weights_only=True)
        digest=hashlib.sha256()
        for module in ("actor","fore"):
            for key,t in sorted(state[module].items()):
                digest.update(key.encode());digest.update(t.contiguous().numpy().tobytes())
        # At step 0 the release and action predictors are shared online.
        signature=(seed,digest.hexdigest(),state["selected"]==0)
        canon=mapping.setdefault(signature,policy)
        rows.append((arm,pref,selector,seed,state["selected"],signature[1],canon))
    pd.DataFrame(rows,columns="arm pref selector seed step sha canonical".split()).to_csv(OUT/"aliases.csv",index=False)
    return sorted((v,k[0]) for k,v in mapping.items())


def replay_window(job):
    date,policies=job
    for name,seed in policies:
        if not (OUT/f"d{date}_{name}_{seed}.csv").exists():
            base.replay(name,seed,date,directory=OUT)
    return date


def initialize(queue):
    os.sched_setaffinity(0,{queue.get()})
    torch.set_num_threads(1)
    threadpool_limits(1)


if __name__=="__main__":
    import multiprocessing as mp
    dates=json.loads((OUT/"lock.json").read_text())["dates"]
    cpu=cores(12)
    ctx=mp.get_context("fork")
    queue=ctx.Queue()
    for c in cpu: queue.put(c)
    with concurrent.futures.ProcessPoolExecutor(len(cpu),mp_context=ctx,initializer=initialize,initargs=(queue,)) as pool:
        classics=[pool.submit(replay_window,(d,[("qp",0),("acopf",0)])) for d in dates]
        tasks=[pool.submit(fit,x) for x in itertools.product("ABC",WEIGHTS,SEEDS)]
        for future in concurrent.futures.as_completed(tasks):
            print("fitted",future.result(),flush=True)
        policies=aliases()
        print("distinct deployed policies",len(policies),flush=True)
        tasks=[pool.submit(replay_window,(d,policies)) for d in dates]
        for future in concurrent.futures.as_completed(tasks+classics):
            print("finished window",future.result(),flush=True)
