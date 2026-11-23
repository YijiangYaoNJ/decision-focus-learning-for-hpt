"""Audit locked checkpoints and summarize date-paired closed-loop tests."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "ieee123_expanded_cuda_v1"
DATA = RUNS / "ieee123_expanded_trajectories_v1"
METRICS = ["kwh", "voltage_s", "vpeak", "overload_s", "opeak", "ms_p50", "ms_p95"]


def main(output=None, source=OUT):
    output = Path(output) if output is not None else RUNS.parents[1] / 'work' / 'tables'
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((source / "protocol.json").read_text())
    assert hashlib.sha256((DATA / "lock.json").read_bytes()).hexdigest() == protocol["dataset_sha"]
    manifest = pd.read_csv(DATA / "manifest.csv")
    choices = json.loads((source / "selection.json").read_text())
    complete = json.loads((source / "complete.json").read_text())
    assert complete["paired_runs"] == len(choices) == 9
    assert set(choices) == {f"{arm}_{seed}" for arm in protocol["arms"] for seed in protocol["seeds"]}
    records = []
    for name, epoch in choices.items():
        folder = source / name
        arm, seed = name.split("_")
        scores = [json.loads(f.read_text()) for f in folder.glob("epoch_*.json")]
        assert min(scores, key=lambda r: (r["trade"], r["epoch"]))["epoch"] == epoch
        assert (folder / f"weights_{epoch}.pt").is_file()
        for split in complete["test_splits"]:
            expected = manifest.loc[manifest.split == split, ["id", "date"]]
            raw = pd.read_csv(folder / f"{split}_{epoch}.csv")
            merged = raw.merge(expected, how="outer", on=["id", "date"], validate="one_to_one", indicator=True)
            assert merged._merge.eq("both").all()
            assert np.isfinite(raw[METRICS].to_numpy()).all() and (raw[METRICS] >= 0).all().all()
            assert (raw[["voltage_s", "overload_s"]] <= 5400).all().all()
            assert raw.issued.between(0, 1080).all()
            assert (raw.ms_p95 >= raw.ms_p50).all()
            records.append(raw.assign(split=split, arm=arm, seed=int(seed)))
    all_rows = pd.concat(records, ignore_index=True)
    date_seed = all_rows.groupby(["split", "arm", "seed", "date"])[METRICS].mean()
    daily = date_seed.groupby(level=["split", "arm", "date"]).mean()
    means = daily.groupby(level=["split", "arm"]).mean()
    means.to_csv(output / "test_summary.csv")
    differences = []
    for split in complete["test_splits"]:
        for comparator in ("mse", "frozen"):
            paired = daily.loc[split, "joint"] - daily.loc[split, comparator]
            paired.assign(split=split, comparator=comparator).to_csv(
                output / f"{split}_joint_minus_{comparator}_by_date.csv")
            seed_diffs = date_seed.loc[split, "joint"] - date_seed.loc[split, comparator]
            seed_diffs.groupby(level="seed").mean().to_csv(output / f"{split}_joint_minus_{comparator}_by_seed.csv")
            rng = np.random.default_rng(123)
            # Descriptive sampling interval on consecutive September dates.
            # Average over the same three seeds first; keep adjacent days together.
            samples = None
            if split == "test_september":
                assert len(paired) == 30 and (np.diff(paired.index) == 1).all()
                starts = rng.integers(0, 30, size=(10000, 10))
                ids = ((starts[:, :, None] + np.arange(3)) % 30).reshape(-1, 30)
                samples = paired.to_numpy()[ids].mean(axis=1)
            for i, metric in enumerate(METRICS):
                bounds = [None, None] if samples is None else np.quantile(samples[:, i], [.025, .975])
                diff = paired[metric]
                differences.append(dict(split=split, comparator=comparator, metric=metric,
                    delta=diff.mean(), dates_lower=int((diff < 0).sum()), dates=len(diff),
                    ci_low=bounds[0], ci_high=bounds[1]))
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    print(means.to_string())
    print(pd.DataFrame(differences).query("comparator=='mse'").to_string(index=False))


if __name__ == "__main__":
    main()
