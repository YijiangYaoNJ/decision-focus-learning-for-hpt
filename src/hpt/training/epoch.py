"""Shared paired streams, plant queries and optimizer updates for the three arms."""

from time import perf_counter

import numpy as np
import torch

from hpt.config import CYCLES, SPEC, TRAJECTORIES
from hpt.control.observations import global_features, local_features
from hpt.data.trajectories import Trajectory, inputs, make_index
from hpt.physics.power_flow import ProposalPowerFlow
from hpt.physics.safety import disk
from hpt.training.feedback import (
    execution_gradient,
    proposal_gradient,
)
from hpt.training.objectives import policy_forecast, training_loss
from hpt.training.storage import record

BATCH = SPEC["batch_size"]
WEIGHT = SPEC["loss"]["risk_weight"]


def paired_batches(dataset, seed, epoch):
    units = iter(dataset.epoch(seed, epoch))
    active = []
    while True:
        while len(active) < BATCH:
            item = next(units, None)
            if item is None:
                break
            obj, indices, last = item
            active.append([obj, iter(indices), last])
        batch = []
        for stream in active:
            index = next(stream[1], None)
            if index is not None:
                batch.append((stream, index))
        if not active:
            return
        active = [s for s, _ in batch]
        if batch:
            yield batch


def fit_epoch(ds, actor, fore, opt, arm, seed, epoch, device, folder):
    def tensor(array):
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)

    directions = np.random.default_rng([seed, epoch, 54])
    pools = [
        (ds.legacy.d["past"][ds.legacy.train], ds.legacy.d["target"][ds.legacy.train])
    ]
    with np.load(TRAJECTORIES / "train.npz") as z:
        for key, r in ds.rows.iterrows():
            idx = make_index(z[key + "_pv"], int(r.slot))
            pools.append((idx["past"], idx["target"]))
    history, targets = [tensor(np.concatenate(v)) for v in zip(*pools)]
    exposures = len(history)
    draw = np.random.default_rng([seed, epoch, 128])
    visited = updates = 0
    total = 0.0
    start = perf_counter()
    for batch in paired_batches(ds, seed, epoch):
        rows = []
        obs = []
        local = []
        histories = []
        for stream, i in batch:
            obj, _, last = stream
            p, h, weather, truth = inputs(obj, i)

            obs.append(global_features(p, h))
            local.append(local_features(p, last, h))
            histories.append(h)
            rows.append((stream, i, p, weather, truth))
        h = tensor(histories)
        path = fore(h)
        cmd = actor(tensor(obs), h, policy_forecast(path, arm), tensor(local))
        raw = disk(cmd.detach().cpu().numpy())
        gradients = []
        for action, (stream, i, p, weather, truth) in zip(raw, rows):
            obj, _, last = stream
            final = next_plant = None
            if isinstance(obj, Trajectory):
                final = i == CYCLES - 1
                if not final and obj.d["slot"][i] != obj.d["slot"][i + 1]:
                    next_plant = obj.plant(i + 1)
            grad, stream[2], cost = execution_gradient(
                obj,
                i,
                last,
                action,
                directions,
                WEIGHT,
                p,
                weather,
                truth,
                True,
                final,
                next_plant,
            )
            safety, _ = proposal_gradient(ProposalPowerFlow(p), action, truth[1:])
            gradients.append(grad + WEIGHT * safety)
            total += cost

        indices = torch.as_tensor(
            draw.choice(exposures, SPEC["supervised_batch_size"], replace=False),
            device=device,
        )
        loss = training_loss(
            cmd, tensor(gradients), arm, fore, history[indices], targets[indices]
        )
        opt.zero_grad()
        loss.backward()
        norms = {}
        for name, network in (("actor", actor), ("fore", fore)):
            norms[name] = float(
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    SPEC["optimizer"]["clip_norm"],
                    error_if_nonfinite=True,
                )
            )
        opt.step()
        visited += len(batch)
        updates += 1
        if updates == 1 or updates % 50 == 0:
            record(
                folder / "progress.json",
                dict(
                    stage="train",
                    arm=arm,
                    seed=seed,
                    epoch=epoch,
                    cycles=visited,
                    updates=updates,
                    training_cycles=exposures,
                    elapsed_s=perf_counter() - start,
                    observed_cost=total / visited,
                    grad_norm=norms,
                    device=torch.cuda.get_device_name(device),
                ),
            )
    assert visited == exposures == SPEC["train_cycles"]
    return updates
