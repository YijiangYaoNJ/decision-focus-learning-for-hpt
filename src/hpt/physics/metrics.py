"""Voltage, converter and winding utilization, and total network + converter losses."""

import json

import numpy as np

from hpt.config import ROOT, SPEC

SITES = json.loads((ROOT / SPEC["service_areas"]).read_text())["sites"]
CAP = np.array([300, 300, 500, 500, 500, 500])


def phase_power(p):
    return np.repeat(p * CAP / 3, 3, axis=-1)


def metrics(model, state, slot, scenario, arm, ratings):
    v = state["v"]
    keep = np.ones(model.original_n, bool)
    keep[model.secondary[0]] = False
    delta = v[model.secondary[0]]
    ll = abs(delta - np.roll(delta, -1)) / 480
    u = np.r_[state["pu"][: model.original_n][keep], ll]
    main, main_loss = [], 0.0
    for j, name in enumerate(model.site_names):
        i = model.sites[name][0]
        ids, mats, _ = model.parts[i]
        terminals = np.r_[v, 0j][ids]
        current = mats[0] @ terminals
        main_loss += (terminals @ current.conj()).real / 1000
        if j == 0:
            # Delta/delta branch windings carry delta currents rather than line currents.
            hv = terminals[:3] - np.roll(terminals[:3], -1)
            lv = terminals[4:7] - np.roll(terminals[4:7], -1)
            z = model.cfg["main_z_pu"][j] * 480**2 / (ratings[j] * 1000)
            iw = (hv * (0.48 / 4.16) - lv) / z
            main.append(
                np.max(np.maximum(abs(hv) * (0.48 / 4.16), abs(lv)) * abs(iw))
                / (1000 * ratings[j])
            )
        else:
            main.append(np.max(abs(terminals * current.conj())) / (1000 * ratings[j]))
    port = state["ports"].max() if "ports" in state else 0.0
    return dict(
        scenario=scenario,
        arm=arm,
        slot=slot,
        day=slot // 96 + 1,
        vmin=u.min(),
        vmax=u.max(),
        excursion=max(0, 0.95 - u.min(), u.max() - 1.05),
        violation=bool(u.min() < 0.95 or u.max() > 1.05),
        main_util=max(main),
        peak_main_site=SITES[int(np.argmax(main))]["element"],
        port_util=port,
        overload=bool(max(max(main), port) > 1 + 1e-7),
        grid_kw=state["grid"].real,
        load_kw=state["load_kw"],
        pv_kw=state["pv_kw"],
        main_kw=main_loss,
        network_kw=state["network_kw"],
        heat_kw=state["heat_kw"],
        loss_kw=state["loss_kw"],
    )


def measure(model, plant, a, power, mode):
    try:
        state = model.solve(
            plant.slot,
            a,
            "high",
            plant.taps,
            generation_kw=power,
            tolerance=1e-8,
            max_iterations=250,
        )
    except ArithmeticError:
        state = model.solve(
            plant.slot,
            a,
            "high",
            plant.taps,
            generation_kw=power,
            tolerance=1e-8,
            max_iterations=1500,
        )
    return state, metrics(model, state, plant.slot, "high", mode, model.ratings.ravel())


def scores(model, plant, old, new, delay, power, mode):
    ms = [
        measure(model, plant, new if new is not None and s >= delay else old, p, mode)[
            1
        ]
        for s, p in enumerate(power)
    ]
    return dict(
        kwh=sum(m["loss_kw"] for m in ms) / 3600,
        voltage_s=sum(m["violation"] for m in ms),
        overload_s=sum(m["overload"] for m in ms),
        v_peak=max(m["excursion"] for m in ms),
        overload_peak=max(
            0.0, max(max(m["main_util"], m["port_util"]) for m in ms) - 1
        ),
    )
