"""Run date-paired jobs on distinct physical cores and resume completed units."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SCRIPT=Path(__file__).with_name("ieee123_direct_confirm.py")
OUT=SCRIPT.parents[1]/"runs/ieee123_direct_confirm_v1"
SEEDS=(11,23,37)
LEARNED=("frozen","mse","joint")


def run(args,core):
    tag="_".join(map(str,args))
    target=(OUT/f"{args[1]}_{args[2]}_updates.csv" if args[0]=="train"
            else OUT/f"d{args[3]}_{args[1]}_{args[2]}.csv")
    if target.exists():
        return
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
    with (OUT/f"{tag}.log").open("w") as log:
        subprocess.run(["taskset","-c",str(core),sys.executable,"-u",str(SCRIPT),*map(str,args)],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print("finished",tag,flush=True)


def cores(limit):
    def read():
        return {int(k[3:]):list(map(int,v.split())) for k,v in
                (r.split(maxsplit=1) for r in Path("/proc/stat").read_text().splitlines() if r.startswith("cpu") and r[3:4].isdigit())}
    first=read();time.sleep(1);last=read()
    unique=set();usable=[]
    for c in sorted(os.sched_getaffinity(0)):
        path=Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        siblings=(path/"thread_siblings_list").read_text().strip()
        if siblings in unique:
            continue
        unique.add(siblings)
        group=[i for i in os.sched_getaffinity(0) if (Path(f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list").read_text().strip()==siblings)]
        idle=min((last[i][3]-first[i][3])/max(1,sum(last[i][:8])-sum(first[i][:8])) for i in group)
        usable.append((idle,c))
    return [c for idle,c in sorted(usable,reverse=True)[:limit]]


if __name__=="__main__":
    dates=json.loads((OUT/"lock.json").read_text())["dates"]
    cpus=cores(len(dates)+len(SEEDS)*len(LEARNED))
    assert len(cpus)==len(dates)+9
    print("CPU assignment",cpus,flush=True)
    def classic(i,date):
        for arm in ("qp","acopf"):
            run(("replay",arm,0,date),cpus[i])
    def learned(i,date):
        for seed in SEEDS:
            for arm in LEARNED:
                run(("replay",arm,seed,date),cpus[i])
    with concurrent.futures.ThreadPoolExecutor(len(cpus)) as pool:
        training=[pool.submit(run,("train",arm,seed),cpus[len(dates)+i]) for i,(arm,seed) in
                  enumerate((a,s) for s in SEEDS for a in LEARNED)]
        baselines=[pool.submit(classic,i,d) for i,d in enumerate(dates)]
        for job in concurrent.futures.as_completed(training+baselines):
            job.result()
        tests=[pool.submit(learned,i,d) for i,d in enumerate(dates)]
        for job in concurrent.futures.as_completed(tests):
            job.result()
