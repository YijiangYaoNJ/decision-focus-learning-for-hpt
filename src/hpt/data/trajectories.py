"""Load locked data partitions and carry the issued command through chronological windows."""

import json

import numpy as np
import pandas as pd
import torch

from hpt.config import CYCLES, DATA, MANIFEST, SPEC, sha
from hpt.config import TRAJECTORIES as OUT
from hpt.forecasting.solar import forecast_net, tensor
from hpt.physics.ieee123 import plant_model
from hpt.physics.metrics import phase_power
from hpt.physics.power_flow import BatchAC

RAW = DATA / "legacy/cases.npz"


def read_manifest():
    frame = pd.read_csv(MANIFEST)
    for split, group in frame.groupby("split"):
        ranges = SPEC["date_ranges"][split]
        if not group.date.map(
            lambda day: any(start <= day <= end for start, end in ranges)
        ).all():
            raise ValueError(f"Dates outside configured {split} range")
    return frame


class LegacyCases:
    def plant(self, i):
        key = (int(self.d["slot"][i]), tuple(self.d["taps"][i]))
        if key not in self.plants:
            self.plants[key] = BatchAC(self.model, *key)
        return self.plants[key]


def study_data():
    data = LegacyCases()
    if (
        sha(RAW)
        != json.loads((RAW.parent / "lock.json").read_text())["files"]["cases.npz"]
    ):
        raise ValueError(RAW)
    data.d = dict(np.load(RAW))
    valid = np.isfinite(data.d["target"]).all(axis=(1, 2))
    data.val = np.flatnonzero(
        valid & (data.d["date"] > SPEC["date_ranges"]["train"][0][1])
    )
    data.train = np.flatnonzero(
        valid & (data.d["date"] <= SPEC["date_ranges"]["train"][0][1])
    )
    data.starts = np.arange(len(valid) - len(data.d["applied"]), len(valid), 120)
    assert len(data.val) == 36 and len(data.train) == 5254
    assert all(np.all(np.isin(np.arange(i, i + 120), data.train)) for i in data.starts)
    data.model, data.plants = plant_model(), {}
    with torch.no_grad():
        data.ref = forecast_net()(tensor(data.d["past"])).numpy()
    return data


def make_index(pv, slot0):
    samples = (
        np.lib.stride_tricks.sliding_window_view(pv, 16, axis=0)[::5]
        .swapaxes(1, 2)
        .copy()
    )
    assert samples.shape == (CYCLES, 16, 6)
    return dict(
        past=samples[:, :11],
        truth=samples[:, 10:15],
        target=samples[:, 11:16],
        slot=slot0 + np.arange(CYCLES) * 5 // 900,
    )


class Trajectory:
    """One chronological unit for physics-feedback training, or for evaluation with locked weights."""

    def __init__(self, pv, slot, taps, initial, model, ref_net):
        self.d = make_index(pv, slot)
        self.initial = initial.copy()
        self.taps = tuple(taps)
        self.model = model
        self.plants = {}
        with torch.inference_mode():
            self.ref = ref_net(tensor(self.d["past"].copy())).numpy()

    def plant(self, i):
        slot = int(self.d["slot"][i])
        if slot not in self.plants:
            self.plants[slot] = BatchAC(self.model, slot, self.taps)
        return self.plants[slot]


class ExpandedTrainingSet:
    """Original training pool plus continuous extensions. Never opens validation or test archives."""

    def __init__(self):
        lock = json.loads((OUT / "lock.json").read_text())
        for name in ("manifest.csv", "train.npz"):
            if sha(OUT / name) != lock["files"][name]:
                raise ValueError(name)
        if sha(RAW) != lock["legacy_sha"]:
            raise ValueError(RAW)
        self.legacy = study_data()
        self.rows = read_manifest().query("split=='train'").set_index("id")
        self.ref_net = forecast_net()

    def epoch(self, seed, epoch):
        """Visit every legacy singleton/chain and every new chain once; carry the returned action within each."""
        d = self.legacy
        chains = [np.arange(s, s + 120) for s in d.starts]
        single = np.setdiff1d(d.train, np.concatenate(chains))
        units = (
            [("legacy", c) for c in chains]
            + [("legacy", np.array([i])) for i in single]
            + [("expanded", k) for k in self.rows.index]
        )
        rng = np.random.default_rng([seed, epoch])
        with np.load(OUT / "train.npz") as z:
            for j in rng.permutation(len(units)):
                kind, key = units[j]
                if kind == "legacy":
                    yield d, key, d.d["old"][key[0]].copy()
                else:
                    r = self.rows.loc[key]
                    data = Trajectory(
                        z[key + "_pv"],
                        int(r.slot),
                        z[key + "_taps"],
                        z[key + "_initial"],
                        d.model,
                        self.ref_net,
                    )
                    yield data, range(CYCLES), data.initial.copy()


def evaluation_trajectories(split):
    if split not in ("validation", "test_may", "test_september"):
        raise ValueError(split)
    lock = json.loads((OUT / "lock.json").read_text())
    for name in ("manifest.csv", split + ".npz"):
        if sha(OUT / name) != lock["files"][name]:
            raise ValueError(name)
    frame = read_manifest()
    model = plant_model()
    ref = forecast_net()
    with np.load(OUT / (split + ".npz")) as z:
        for r in frame[frame.split == split].itertuples():
            yield (
                r,
                Trajectory(
                    z[r.id + "_pv"],
                    r.slot,
                    z[r.id + "_taps"],
                    z[r.id + "_initial"],
                    model,
                    ref,
                ),
            )


def inputs(data, i):
    history = data.d["past"][i]
    weather = phase_power(np.vstack((history[-1], data.ref[i])))
    return data.plant(i), history, weather, phase_power(data.d["truth"][i])
