"""Causal global feeder and per-device observations for the action network."""

import numpy as np

from hpt.physics.metrics import phase_power


def feature(p):
    return np.r_[p[-1], 3 * (p[-1] - p[-6]), 1.5 * (p[-6] - p[-11])]


def features(m, slot, taps, past):
    load = m.series[slot].sum()
    return np.r_[
        feature(past),
        load.real / 1000,
        load.imag / 1000,
        np.asarray(taps) / 16,
        np.sin(2 * np.pi * slot / 96),
        np.cos(2 * np.pi * slot / 96),
    ]


def local_features(plant, old, past):
    m = plant.m
    pv = phase_power(past[-1])
    state = m.solve(
        plant.slot,
        old,
        "high",
        plant.taps,
        generation_kw=pv,
        tolerance=1e-8,
        max_iterations=1500,
    )
    v = state["v"][m.secondary]
    pu = np.abs(v) / m.bases[m.secondary]
    pu[0] = np.abs(v[0] - np.roll(v[0], -1)) / 480
    # Per-phase port loading uses the actual series/shunt converter powers.
    spare = 1 - state["ports"].max(axis=0)
    return np.column_stack((old.reshape(6, 9), (pu - 1) * 20, spare))


def global_features(plant, history):
    load = plant.m.series[plant.slot]
    injections = np.column_stack((load.real, load.imag)).ravel() / 100
    return np.r_[features(plant.m, plant.slot, plant.taps, history), injections]
