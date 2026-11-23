"""Chronological closed-loop replay and validation checkpoint scoring."""

import copy

import numpy as np
import pandas as pd
import torch

from hpt.config import SPEC
from hpt.control.execution import IEEE123Feeder, NeuralController, issue
from hpt.control.interfaces import Controller, Feeder
from hpt.data.trajectories import evaluation_trajectories
from hpt.forecasting.solar import forecast_net, tensor


def replay_window(
    trajectory,
    controller: Controller,
    feeder: Feeder,
    predict,
    reference,
    *,
    observe=None,
):
    last = trajectory.initial
    delays = []
    energy = voltage_s = overload_s = peak = overload_peak = issued_count = 0
    for i, history in enumerate(trajectory.d["past"]):
        frame = trajectory.plant(i)
        issued, status, delay, proposal = issue(
            controller, feeder, predict, reference, frame, last, history
        )
        metrics = feeder.score(frame, last, issued, delay, trajectory.d["truth"][i])
        if observe is not None:
            observe(i, proposal, issued, metrics)
        if issued is not None:
            last = issued
        delays.append(delay * 1000)
        issued_count += status == "issued"
        energy += metrics["kwh"]
        voltage_s += metrics["voltage_s"]
        overload_s += metrics["overload_s"]
        peak = max(peak, metrics["v_peak"])
        overload_peak = max(overload_peak, metrics["overload_peak"])
    return dict(
        kwh=energy,
        voltage_s=voltage_s,
        overload_s=overload_s,
        vpeak=peak,
        opeak=overload_peak,
        issued=issued_count,
        ms_p50=np.median(delays),
        ms_p95=np.quantile(delays, 0.95),
    )


@torch.inference_mode()
def evaluate(actor, fore, split, folder, epoch, *, max_windows=None, observe=None):
    controller = NeuralController(copy.deepcopy(actor).cpu().eval())
    fore = copy.deepcopy(fore).cpu().eval()
    ref = forecast_net()
    feeder = IEEE123Feeder()

    def predict(h):
        return fore(tensor(h[None])).numpy()[0]

    def reference(h):
        return ref(tensor(h[None])).numpy()[0]

    rows = []
    for r, tr in evaluation_trajectories(split):
        row = replay_window(tr, controller, feeder, predict, reference, observe=observe)
        rows.append(dict(id=r.id, date=r.date, **row))
        if max_windows is not None and len(rows) >= max_windows:
            break
    f = pd.DataFrame(rows)
    f.to_csv(folder / f"{split}_{epoch}.csv", index=False)
    m = f.groupby("date").mean(numeric_only=True).mean()
    duration = SPEC["window_seconds"]
    freq_v, peak_v, freq_o, peak_o = SPEC["loss"]["risk_coefficients"]
    risk = (
        freq_v * m.voltage_s / duration
        + peak_v * m.vpeak
        + freq_o * m.overload_s / duration
        + peak_o * m.opeak
    )
    return dict(
        trade=float(
            m.kwh / (duration / 3600) / 100 + SPEC["loss"]["risk_weight"] * risk
        ),
        kwh=m.kwh,
        voltage_s=m.voltage_s,
        overload_s=m.overload_s,
    )
