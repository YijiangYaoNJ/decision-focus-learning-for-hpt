"""Compare released checkpoint behavior with the independent v0.1.0 oracle."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hpt.config import CHECKPOINTS, REFERENCE, SPEC
from hpt.control.policy import networks
from hpt.data.trajectories import ExpandedTrainingSet, inputs
from hpt.evaluation.replay import evaluate
from hpt.forecasting.solar import tensor
from hpt.physics.power_flow import ProposalPowerFlow
from hpt.physics.safety import disk
from hpt.training.feedback import (
    execution_gradient,
    proposal_gradient,
)
from hpt.training.objectives import policy_forecast, training_loss


def main(oracle, output):
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    ds = ExpandedTrainingSet()
    choices = json.loads((REFERENCE / "selection.json").read_text())
    maxima = {}
    nonzero = {}

    def compare(label, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(
            actual, expected, atol=1e-7, rtol=2e-6, equal_nan=True, err_msg=label
        )
        diff = np.abs(actual - expected)
        maxima[label] = max(maxima.get(label, 0.0), float(np.nanmax(diff, initial=0)))
        nonzero[label] = nonzero.get(label, False) or bool(
            np.any(np.nan_to_num(expected) != 0)
        )

    with np.load(oracle / "steps.npz") as archive:
        for seed in SPEC["seeds"]:
            for arm in SPEC["arms"]:
                key = f"{arm}_{seed}"
                a, f = networks(seed, ds.legacy.model)
                ck = torch.load(
                    CHECKPOINTS / "selected" / key / f"weights_{choices[key]}.pt",
                    weights_only=True,
                    map_location="cpu",
                )
                a.load_state_dict(ck["actor"])
                f.load_state_dict(ck["fore"])
                f.requires_grad_(arm != "frozen")
                for tag, i in enumerate(ds.legacy.train[[0, 150, 982, -1]]):
                    for module in (a, f):
                        module.zero_grad(set_to_none=True)
                    p, h, w, truth = inputs(ds.legacy, i)
                    last = ds.legacy.d["old"][i]
                    past = tensor(h[None])
                    path = f(past)
                    cmd = a.suggest(p, last, h, policy_forecast(path, arm))
                    action = disk(cmd.detach().numpy())[0]
                    grad, end, cost = execution_gradient(
                        ds.legacy,
                        i,
                        last,
                        action,
                        np.random.default_rng(tag),
                        4.0,
                        p,
                        w,
                        truth,
                        True,
                        None,
                        None,
                    )
                    safety, _ = proposal_gradient(
                        ProposalPowerFlow(p), action, truth[1:]
                    )
                    loss = training_loss(
                        cmd,
                        tensor(grad + 4 * safety),
                        arm,
                        f,
                        past,
                        tensor(ds.legacy.d["target"][i : i + 1]),
                    )
                    loss.backward()
                    row = dict(
                        pred=path.detach().numpy(),
                        action=action,
                        end=end,
                        execution=grad,
                        safety=safety,
                        cost=cost,
                    )
                    for label, module in (("actor", a), ("forecast", f)):
                        row[label] = torch.cat(
                            [
                                (
                                    v.grad
                                    if v.grad is not None
                                    else torch.zeros_like(v)
                                ).flatten()
                                for v in module.parameters()
                            ]
                        ).numpy()
                    for k, v in row.items():
                        compare(k, v, archive[f"{key}/{tag}/{k}"])
    assert all(nonzero[k] for k in ("execution", "safety", "actor", "forecast")), (
        nonzero
    )
    proposed = []
    issued = []
    metrics = []

    def observe(i, p, c, m):
        proposed.append(p.copy())
        issued.append(c.copy() if c is not None else np.full(54, np.nan))
        metrics.append(list(m.values()))

    a, f = networks(11, ds.legacy.model)
    epoch = choices["joint_11"]
    ck = torch.load(
        CHECKPOINTS / "selected/joint_11" / f"weights_{epoch}.pt",
        weights_only=True,
        map_location="cpu",
    )
    a.load_state_dict(ck["actor"])
    f.load_state_dict(ck["fore"])
    evaluate(a, f, "test_may", output, epoch, max_windows=1, observe=observe)
    with np.load(oracle / "trajectory.npz") as archive:
        for name, values in (
            ("proposed", proposed),
            ("issued", issued),
            ("metrics", metrics),
        ):
            compare(name, values, archive[name])
    result = dict(
        matched_cases=36,
        closed_loop_cycles=len(metrics),
        issued=int(np.isfinite(issued).all(axis=1).sum()),
        max_abs_difference=maxima,
    )
    print(json.dumps(result, indent=2))
    (output / "parity.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.oracle, args.output)
