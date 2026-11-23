"""Lock new August windows and export the verified feeder for identical replay on Linux."""
import gzip
import json
import pickle
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

import ieee123_matched_holdout as previous
from ieee123_six import SixHPT

OUT = previous.ROOT / "runs/ieee123_direct_confirm_v1"


def plant_model():
    lock = json.loads((OUT / "lock.json").read_text())
    file = OUT / "feeder.pkl.gz"
    assert previous.joint.sha(file) == lock["feeder"]
    with gzip.open(file, "rb") as src:
        model = SixHPT.__new__(SixHPT)
        model.__dict__.update(pickle.load(src))
    model.factor = lru_cache(maxsize=32)(model._factor)
    return model


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "lock.json").exists():
        raise FileExistsError("Windows are already locked")
    sys.path.insert(0, str(previous.ROOT / "src"))
    used = set(pd.read_csv(previous.OUT / "windows.csv").date)
    # Actual evaluation rows only. Eligibility lists also name unused dates.
    for file in (previous.ROOT / "runs").glob("ieee123*/**/*.csv"):
        if file.stat().st_size > 64_000_000:
            continue
        columns = pd.read_csv(file, nrows=0).columns
        if "date" in columns:
            dates = pd.read_csv(file, usecols=["date"])["date"].astype(str).str.replace("-", "", regex=False)
            used.update(int(d) for d in dates.unique() if len(d)==8 and d.startswith("201008") and d.isdigit())
    draws = previous.choose(previous.ARCHIVE, 1, np.random.default_rng(111201))
    eligible = [r for r in draws if r[0] not in used]
    assert len(eligible) >= 12
    rng = np.random.default_rng(111202)
    selected = [eligible[rng.choice(block)] for block in np.array_split(np.arange(len(eligible)), 12)]
    forecasts, _ = previous.predictor(np.stack([r[3] for r in selected]))
    _, model, *_ = previous.engine.build()
    fields = dict(vars(model))
    del fields["factor"]
    with gzip.open(OUT / "feeder.pkl.gz", "wb", compresslevel=2) as dest:
        pickle.dump(fields, dest, protocol=4)
    rows, arrays = [], {}
    for (date, second, slot, _, pv), forecast in zip(selected, forecasts):
        st = model.step(slot, previous.ZERO, "high", generation_kw=previous.phase_power(forecast))
        initial = previous.local(st, model, previous.ZERO)
        arrays.update({f"{date}_pv":pv, f"{date}_taps":st["taps"], f"{date}_initial":initial})
        rows.append((date,second,slot))
    np.savez_compressed(OUT/"windows.npz", **arrays)
    pd.DataFrame(rows, columns=["date","second","slot"]).to_csv(OUT/"windows.csv", index=False)
    digest = previous.joint.sha
    lock = dict(excluded_august=sorted(used), dates=[r[0] for r in rows], seeds=[11,23,37],
                archive=digest(previous.ARCHIVE), feeder=digest(OUT/"feeder.pkl.gz"),
                manifest=digest(OUT/"windows.csv"), windows=digest(OUT/"windows.npz"),
                train="April through 22nd; April validation thereafter; paired step-0 risk caps; patience 40, max 120",
                arms=["frozen","mse","joint","qp","acopf"], primary="joint minus mse. Average seeds within date first; date bootstrap 95% intervals. Require no increase in voltage/overload duration or peak, and lower energy.",
                rules="Direct issuance only, common frozen seed-11 screening path, rejection or 5 s timeout holds last. Measured compute delay included. AC-OPF uses a mean forecast snapshot; six-frame acceptance for all arms.")
    (OUT/"lock.json").write_text(json.dumps(lock,indent=2))
    restored = plant_model()
    d,_,slot = rows[0]
    for obj in [model,restored]:
        state=obj.solve(slot, arrays[f"{d}_initial"], "high", tuple(arrays[f"{d}_taps"]), generation_kw=previous.phase_power(arrays[f"{d}_pv"][10]))
        if obj is model:
            original = state["v"]
        else:
            np.testing.assert_array_equal(original, state["v"])
    print("Locked dates", lock["dates"], flush=True)


if __name__=="__main__":
    prepare()
