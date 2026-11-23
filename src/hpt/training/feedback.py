"""Physical feedback for issued commands and proposal safety before screening."""

import numpy as np

from hpt.config import SPEC
from hpt.physics.metrics import phase_power
from hpt.physics.safety import disk, outcomes

RISK = np.array(SPEC["loss"]["risk_coefficients"])
DIRECTIONS = 4
EPSILON = 0.002


def ac(plant, action, power):
    try:
        return plant.evaluate(action, power)
    except ArithmeticError:
        return plant.evaluate(action, power, max_iterations=1500)


def components(plant, y):
    n = 2 * plant.m.original_n
    voltage = -y[:, 1 : 1 + n].min(1)
    capacity = np.sqrt(np.maximum(0.0, 1 - y[:, 1 + n : -18].min(1))) - 1
    return voltage.clip(0), capacity.clip(0)


def penalty(plant, y):
    v, s = components(plant, y)
    return 0.1 * (v / 0.003 + s / 0.01)


def proposal_gradient(plant, x, truth):
    # Columns reflect the actual actuator model (including its disk saturation).
    # Central differences of the physical residual; use the active subgradient at
    # the base point rather than differencing the hinge across a change of sign.
    h = 2e-5
    basis = h * np.eye(54)
    probes = np.vstack((x + basis, x - basis))
    nv = 2 * plant.m.original_n
    grad = np.zeros(54)
    ys = ac(plant, x, truth)
    for pv, y in zip(truth, ys):
        vrow = 1 + int(np.argmin(y[1 : 1 + nv]))
        srow = 1 + nv + int(np.argmin(y[1 + nv : -18]))
        util = np.sqrt(max(0.0, 1 - y[srow]))
        if y[vrow] >= 0 and util <= 1:
            continue
        z = ac(plant, probes, pv)
        columns = z[:, [vrow, srow]]
        jac = (columns[:54] - columns[54:]) / (2 * h)
        if y[vrow] < 0:
            grad -= 0.1 / 0.003 * jac[:, 0]
        if util > 1:
            grad -= 0.1 / 0.01 * jac[:, 1] / (2 * util)
    return grad / len(truth), penalty(plant, ys).mean()


def execution_gradient(
    data,
    i,
    last,
    raw,
    rng,
    w,
    plant,
    weather,
    truth,
    applied_interval,
    final_cycle,
    last_frame_plant,
):
    dirs = rng.choice([-1.0, 1.0], (DIRECTIONS, 54))
    delta = dirs * EPSILON
    bank = disk(np.vstack((raw, raw + delta, raw - delta)))
    execution_truth = truth
    if applied_interval:
        execution_truth = phase_power(data.d["target"][i])
        if final_cycle is None:
            final_cycle = i >= data.starts[0] and (i - data.starts[0] + 1) % 120 == 0
        if final_cycle:
            execution_truth = execution_truth[:4]
    end, kw, risk, _ = outcomes(
        plant,
        last,
        bank,
        weather,
        execution_truth,
        first_second_held=not applied_interval,
        last_frame_plant=last_frame_plant,
    )
    values = kw / 100 + w * (risk @ RISK)
    grad = (
        (values[1 : 1 + len(dirs)] - values[1 + len(dirs) :])
        @ dirs
        / (2 * EPSILON * len(dirs))
    )
    return grad, end[0], values[0]
