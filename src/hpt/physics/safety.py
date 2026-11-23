"""Forecast feasibility screen, cold boundary check and realized action outcomes."""

import numpy as np

from hpt.physics.metrics import metrics
from hpt.physics.power_flow import Snapshot

MARGIN = 0.003


def disk(x):
    a = np.clip(np.asarray(x, dtype=float).reshape(-1, 18, 3), -1, 1).copy()
    uv = a[:, :, :2]
    uv /= np.maximum(1, np.linalg.norm(uv, axis=2, keepdims=True))
    return a.reshape(-1, 54)


def eval_rows(plant, actions, powers):
    try:
        return plant.evaluate(actions, powers)
    except ArithmeticError:
        return np.concatenate(
            [plant.evaluate(a, g, max_iterations=1500) for a, g in zip(actions, powers)]
        )


def scores(plant, commands, weather):
    """Batched forecast screen plus cold check, using the deployment numerical tolerances."""
    n = 2 * plant.m.original_n
    powers = np.tile(weather, (len(commands), 1))
    actions = np.repeat(commands, len(weather), axis=0)
    y = eval_rows(plant, actions, powers).reshape(len(commands), len(weather), -1)
    out = []
    for x, frames in zip(commands, y):
        c = frames[:, 1:].min(0)
        physical = c.min() >= 0

        def near(raw):
            return (
                raw[:n].min() >= 0
                and raw[-54:].min() >= 0
                and 1 - (1 + 1e-7) ** 2 <= raw[n:-54].min() < 0
            )

        if physical:
            raw = Snapshot(plant.m, plant.slot, plant.taps, weather[0]).evaluate(x)[0][
                1:
            ]
            physical = raw.min() >= 0
            boundary = not physical and near(raw)
        else:
            boundary = near(c)
        if boundary:
            physical = True
            for g in np.unique(weather, axis=0):
                try:
                    st = plant.m.solve(
                        plant.slot,
                        x,
                        "high",
                        plant.taps,
                        generation_kw=g,
                        tolerance=1e-10,
                        max_iterations=1500,
                    )
                except ArithmeticError:
                    physical = False
                    break
                m = metrics(
                    plant.m, st, plant.slot, "high", "train", plant.m.ratings.ravel()
                )
                if not (
                    0.95 <= m["vmin"]
                    and m["vmax"] <= 1.05
                    and m["main_util"] <= 1 + 1e-7
                    and m["port_util"] <= 1
                ):
                    physical = False
                    break
        out.append(
            (max(0.0, MARGIN - c[:n].min()), frames[:, 0].mean()) if physical else None
        )
    return out


def accepted_commands(plant, commands, weather):
    # scores applies the established AC capability tests; only pass/fail is used.
    valid = np.isfinite(commands).all(axis=1) & (np.abs(commands) <= 1).all(axis=1)
    accepted = np.zeros(len(commands), dtype=bool)
    if valid.any():
        try:
            checked = scores(plant, commands[valid], weather)
            accepted[valid] = [r is not None for r in checked]
        except ArithmeticError:
            for i in np.flatnonzero(valid):
                try:
                    accepted[i] = (
                        scores(plant, commands[i : i + 1], weather)[0] is not None
                    )
                except ArithmeticError:
                    pass
    return accepted


def outcomes(
    plant,
    last,
    commands,
    weather,
    truth,
    *,
    first_second_held=True,
    last_frame_plant=None,
):
    accepted = accepted_commands(plant, commands, weather)
    end = np.where(accepted[:, None], commands, last)
    # One-second actuation lag in offline training, measured release times in replay.
    unique, indices = np.unique(end, axis=0, return_inverse=True)
    same_slot = truth if last_frame_plant is None else truth[:-1]
    seconds = len(same_slot)
    y = eval_rows(
        plant, np.repeat(unique, seconds, axis=0), np.tile(same_slot, (len(unique), 1))
    )
    frames = y.reshape(len(unique), seconds, -1)
    if last_frame_plant is not None:
        tail = eval_rows(
            last_frame_plant, unique, np.repeat(truth[-1:], len(unique), axis=0)
        )
        frames = np.concatenate((frames, tail[:, None, :]), axis=1)
    actual = frames[indices]
    if first_second_held:
        actual[:, 0] = eval_rows(plant, last[None], truth[:1])[0]
    n = 2 * plant.m.original_n
    gap = np.maximum(0, -actual[:, :, 1 : 1 + n].min(2))
    util = np.sqrt(np.maximum(0, 1 - actual[:, :, 1 + n : -18].min(2)))
    risk = np.column_stack(
        (
            (gap > 0).mean(1),
            gap.max(1),
            (util > 1 + 1e-7).mean(1),
            (util - 1).clip(0).max(1),
        )
    )
    return end, actual[:, :, 0].mean(1), risk, accepted
