"""Linux CUDA launch, paired early stopping, recovery and test dispatch."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import run

import numpy as np
import torch

from hpt.config import RUNS, SPEC, TRAJECTORIES, sha
from hpt.control.policy import networks
from hpt.data.trajectories import ExpandedTrainingSet
from hpt.evaluation.replay import evaluate
from hpt.physics.ieee123 import plant_model
from hpt.training.epoch import BATCH, WEIGHT, fit_epoch
from hpt.training.storage import atomic_save, record

OUT = Path(os.environ.get("HPT_TRAIN_OUTPUT", RUNS / "retrain"))
ARMS = SPEC["arms"]
SEEDS = SPEC["seeds"]


def worker(arm, seed, epoch):
    folder = OUT / f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    result = folder / f"epoch_{epoch}.json"
    if result.exists():
        return
    ds = ExpandedTrainingSet()
    actor, fore = networks(seed, ds.legacy.model)
    device = torch.device("cuda:0")
    actor.to(device)
    fore.to(device).requires_grad_(arm != "frozen")
    opt = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": SPEC["optimizer"]["actor_lr"]},
            {"params": fore.parameters(), "lr": SPEC["optimizer"]["forecast_lr"]},
        ]
    )

    def snapshot(n):
        return folder / f"weights_{n}.pt"

    if epoch == 1 and not snapshot(0).exists():
        atomic_save(
            dict(
                actor=actor.state_dict(), fore=fore.state_dict(), opt=opt.state_dict()
            ),
            snapshot(0),
        )
    if not snapshot(epoch).exists():
        s = torch.load(snapshot(epoch - 1), map_location=device, weights_only=True)
        actor.load_state_dict(s["actor"])
        fore.load_state_dict(s["fore"])
        opt.load_state_dict(s["opt"])
        updates = fit_epoch(ds, actor, fore, opt, arm, seed, epoch, device, folder)
        atomic_save(
            dict(
                actor=actor.state_dict(),
                fore=fore.state_dict(),
                opt=opt.state_dict(),
                updates=updates,
            ),
            snapshot(epoch),
        )
    for n in [0, 1] if epoch == 1 else [epoch]:
        if (folder / f"epoch_{n}.json").exists():
            continue
        s = torch.load(snapshot(n), map_location=device, weights_only=True)
        actor.load_state_dict(s["actor"])
        fore.load_state_dict(s["fore"])
        val = evaluate(actor, fore, "validation", folder, n)
        record(
            folder / f"epoch_{n}.json",
            dict(epoch=n, updates=s.get("updates", 0), **val),
        )


def launch(gpus):
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (OUT / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = dict(
        dataset_sha=sha(TRAJECTORIES / "lock.json"),
        arms=ARMS,
        seeds=SEEDS,
        weight=WEIGHT,
        config=SPEC,
        batch=BATCH,
        selector="Lowest date-balanced validation trade, including initialization; earliest tie.",
        deployment="Identical timed CPU action issue and AC replay, training networks on CUDA.",
    )
    if (OUT / "protocol.json").exists():
        assert json.loads((OUT / "protocol.json").read_text()) == json.loads(
            json.dumps(protocol)
        )
    record(OUT / "protocol.json", protocol)

    def job(args):
        arm, seed, stage, epoch = args
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpus[SEEDS.index(seed) % len(gpus)])
        )
        with (OUT / f"{arm}_{seed}.log").open("a") as log:
            run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "hpt.training.runner",
                    stage,
                    arm,
                    str(seed),
                    str(epoch),
                ],
                env=env,
                stdout=log,
                stderr=log,
                check=True,
            )

    active = set(SEEDS)
    picked = {}
    with ThreadPoolExecutor(max_workers=9) as pool:
        for epoch in range(1, SPEC["selection"]["max_epochs"] + 1):
            list(
                pool.map(
                    job, [(a, s, "worker", epoch) for s in sorted(active) for a in ARMS]
                )
            )
            for seed in sorted(active):
                improved = False
                for arm in ARMS:
                    folder = OUT / f"{arm}_{seed}"
                    costs = [
                        json.loads((folder / f"epoch_{i}.json").read_text())["trade"]
                        for i in range(epoch + 1)
                    ]
                    picked[arm, seed] = int(np.argmin(costs))
                    improved |= (
                        costs[-1]
                        < min(costs[:-1]) - SPEC["selection"]["min_improvement"]
                    )
                if epoch >= SPEC["selection"]["min_epochs"] and (
                    not improved or epoch == SPEC["selection"]["max_epochs"]
                ):
                    active.remove(seed)
            if not active:
                break
        record(OUT / "selection.json", {f"{a}_{s}": n for (a, s), n in picked.items()})
        list(pool.map(job, [(a, s, "test", n) for (a, s), n in picked.items()]))
    record(
        OUT / "complete.json",
        {"paired_runs": len(picked), "test_splits": ["test_may", "test_september"]},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["launch", "worker", "test"])
    p.add_argument("params", nargs="*")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == "launch":
        launch([int(i) for i in args.params])
    else:
        arm, seed, epoch = args.params
        seed, epoch = int(seed), int(epoch)
        if args.stage == "worker":
            worker(arm, seed, epoch)
        else:
            folder = OUT / f"{arm}_{seed}"
            actor, fore = networks(seed, plant_model())
            state = torch.load(
                folder / f"weights_{epoch}.pt", map_location="cpu", weights_only=True
            )
            actor.load_state_dict(state["actor"])
            fore.load_state_dict(state["fore"])
            for split in ("test_may", "test_september"):
                evaluate(actor, fore, split, folder, epoch)
