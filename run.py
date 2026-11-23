"""Public entry points for the locked IEEE123 six-HPT comparison."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from hpt.config import CHECKPOINTS, REFERENCE, SPEC
from hpt.config import RUNS as WORK


def verify():
    for line in (ROOT / "checksums.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        with (ROOT / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    import pandas as pd

    from hpt.evaluation import tables as results

    output = WORK / "tables"
    results.main(output)
    for source in REFERENCE.glob("*.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(source),
            pd.read_csv(output / source.name),
            check_exact=False,
            rtol=1e-9,
            atol=1e-10,
        )
    print("Verified the released artifacts and regenerated all ten result tables.")


def replay(args):
    import torch

    from hpt.control.policy import networks
    from hpt.evaluation.replay import evaluate
    from hpt.physics.ieee123 import plant_model

    torch.set_num_threads(1)
    name = f"{args.arm}_{args.seed}"
    epoch = json.loads((REFERENCE / "selection.json").read_text())[name]
    actor, fore = networks(args.seed, plant_model())
    saved = torch.load(
        CHECKPOINTS / "selected" / name / f"weights_{epoch}.pt",
        map_location="cpu",
        weights_only=True,
    )
    actor.load_state_dict(saved["actor"])
    fore.load_state_dict(saved["fore"])
    size = "full" if args.windows is None else f"first_{args.windows}"
    output = WORK / "replay" / name / size
    output.mkdir(parents=True, exist_ok=True)
    stats = evaluate(actor, fore, args.split, output, epoch, max_windows=args.windows)
    print(json.dumps(stats, indent=2))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser(
        "verify", help="Check artifacts and reproduce the archived metric tables on CPU"
    )
    rep = subs.add_parser(
        "replay", help="Issue commands with a released checkpoint on a test split"
    )
    rep.add_argument("--arm", choices=SPEC["arms"], default="joint")
    rep.add_argument("--seed", type=int, choices=SPEC["seeds"], default=11)
    rep.add_argument(
        "--split", choices=("test_may", "test_september"), default="test_may"
    )
    rep.add_argument(
        "--windows", type=int, help="Optional positive limit, in manifest order"
    )
    training = subs.add_parser(
        "train", help="Run all paired arms and validation-based stopping on Linux/CUDA"
    )
    training.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "replay":
        if args.windows is not None and args.windows < 1:
            parser.error("--windows must be positive")
        replay(args)
    else:
        if sys.platform != "linux" or any(i < 0 for i in args.gpus):
            parser.error("Training requires Linux and nonnegative CUDA device indices")
        target = WORK / "retrain"
        target.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            HPT_TRAIN_OUTPUT=str(target),
            PYTHONPATH=str(ROOT / "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hpt.training.runner",
                "launch",
                *map(str, args.gpus),
            ],
            env=env,
            check=True,
        )
        from hpt.evaluation import tables as results

        results.main(WORK / "retrained_tables", source=target)


if __name__ == "__main__":
    main()
