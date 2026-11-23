"""Paths and locked protocol for the published six-HPT experiment."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "configs/ieee123.json").read_text())
DATA = ROOT / "data"
TRAJECTORIES = DATA / "trajectories"
MANIFEST = ROOT / SPEC["manifest"]
CHECKPOINTS = ROOT / "checkpoints"
REFERENCE = ROOT / "results/reference"
RUNS = ROOT / "runs"
CYCLES = SPEC["window_seconds"] // SPEC["coordination_seconds"]
DIM = SPEC["hpt_count"] * SPEC["phases"] * SPEC["controls_per_phase"]
if (SPEC["feeder"], DIM, SPEC["coordination_seconds"]) != ("ieee123", 54, 5):
    raise ValueError(
        "The released IEEE123 weights and trajectories require six three-phase HPTs and five-second coordination."
    )
if (
    SPEC["selection"]["metric"] != "date_balanced_trade"
    or SPEC["selection"]["tie"] != "earliest"
):
    raise ValueError("Unsupported checkpoint selection rule")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
