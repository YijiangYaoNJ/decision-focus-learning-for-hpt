"""Generate a numerical oracle from a detached, unmodified v0.1.0 checkout."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkout", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.checkout.resolve() / "hpt_project/tools"))
import ieee123_expanded_train as old

torch.set_num_threads(1)
src = args.checkout / "hpt_project/runs/ieee123_expanded_cuda_v1"
choices = json.loads((src / "selection.json").read_text())
ds = old.data.ExpandedTrainingSet()
arrays = {}
for seed in old.SEEDS:
    for arm in old.ARMS:
        key = f"{arm}_{seed}"
        net, fore = old.base.networks(seed, ds.legacy.model)
        ck = torch.load(
            src / key / f"weights_{choices[key]}.pt",
            weights_only=True,
            map_location="cpu",
        )
        net.load_state_dict(ck["actor"])
        fore.load_state_dict(ck["fore"])
        fore.requires_grad_(arm != "frozen")
        for tag, i in enumerate(ds.legacy.train[[0, 150, 982, -1]]):
            for module in (net, fore):
                module.zero_grad(set_to_none=True)
            obj = ds.legacy
            p, h, weather, truth = old.base.inputs(obj, i)
            last = obj.d["old"][i]
            path = fore(old.base.joint.tensor(h[None]))
            cmd = net.suggest(p, last, h, path if arm == "joint" else path.detach())
            action = old.base.prior.disk(cmd.detach().numpy())[0]
            g, end, cost = old.full.execution_gradient(
                obj,
                i,
                last,
                action,
                np.random.default_rng(tag),
                4.0,
                p,
                weather,
                truth,
                True,
                None,
                None,
            )
            safe, _ = old.full.short.proposal_gradient(
                old.full.ProposalPowerFlow(p), action, truth[1:]
            )
            loss = (cmd * old.base.joint.tensor(g + 4 * safe)).sum()
            if arm != "frozen":
                loss = (
                    loss
                    + 10
                    * (
                        fore(old.base.joint.tensor(h[None]))
                        - old.base.joint.tensor(obj.d["target"][i : i + 1])
                    )
                    .square()
                    .mean()
                )
            loss.backward()
            row = dict(
                pred=path.detach().numpy(),
                action=action,
                end=end,
                execution=g,
                safety=safe,
                cost=cost,
            )
            for label, module in (("actor", net), ("forecast", fore)):
                row[label] = torch.cat(
                    [
                        (
                            v.grad if v.grad is not None else torch.zeros_like(v)
                        ).flatten()
                        for v in module.parameters()
                    ]
                ).numpy()
            arrays.update({f"{key}/{tag}/{k}": v for k, v in row.items()})
np.savez(args.output / "steps.npz", **arrays)
# Observe every issued command and per-cycle physical outcome on the actual old replay path.
original_issue = old.base.issue
original_score = old.data.hourly.scores
proposals, commands, metrics = [], [], []


def issue(*args, **kwargs):
    trace = []
    command, status, delay, extra = original_issue(*args, **kwargs, trace=trace)
    proposals.append(trace[0][1])
    commands.append(command.copy() if command is not None else np.full(54, np.nan))
    return command, status, delay, extra


def scores(*args, **kwargs):
    result = original_score(*args, **kwargs)
    metrics.append(list(result.values()))
    return result


old.base.issue, old.data.hourly.scores = issue, scores
net, fore = old.base.networks(11, ds.legacy.model)
epoch = choices["joint_11"]
ck = torch.load(
    src / "joint_11" / f"weights_{epoch}.pt", weights_only=True, map_location="cpu"
)
net.load_state_dict(ck["actor"])
fore.load_state_dict(ck["fore"])
old.evaluate(net, fore, "test_may", args.output, epoch, max_windows=1)
np.savez(
    args.output / "trajectory.npz", proposed=proposals, issued=commands, metrics=metrics
)
print("Captured 36 matched states and 1,080 closed-loop cycles.")
