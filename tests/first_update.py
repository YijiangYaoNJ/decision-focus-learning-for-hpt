"""Exercise the actual fit_epoch path to its first Adam update, on CPU."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


def capture(checkout, version, filename):
    if version == "old":
        sys.path.insert(0, str(checkout / "hpt_project/tools"))
        import ieee123_expanded_train as original

        dataset = original.data.ExpandedTrainingSet()
        networks, fit_epoch = original.base.networks, original.fit_epoch
    else:
        sys.path.insert(0, str(checkout / "src"))
        from hpt.control.policy import networks
        from hpt.data.trajectories import ExpandedTrainingSet
        from hpt.training.epoch import fit_epoch

        dataset = ExpandedTrainingSet()

    class FinishedUpdate(Exception):
        pass

    def done(optimizer, args, kwargs):
        raise FinishedUpdate

    torch.set_num_threads(1)
    values = {}
    with TemporaryDirectory() as tmp:
        for arm in ("frozen", "mse", "joint"):
            actor, fore = networks(11, dataset.legacy.model)
            fore.requires_grad_(arm != "frozen")
            optimizer = torch.optim.Adam(
                [
                    dict(params=actor.parameters(), lr=0.0005),
                    dict(params=fore.parameters(), lr=0.00015),
                ]
            )
            optimizer.register_step_post_hook(done)
            try:
                fit_epoch(
                    dataset,
                    actor,
                    fore,
                    optimizer,
                    arm,
                    11,
                    1,
                    torch.device("cpu"),
                    Path(tmp),
                )
            except FinishedUpdate:
                pass
            else:
                raise AssertionError("No Adam update observed")
            for name, model in (("actor", actor), ("forecast", fore)):
                for key, parameter in model.state_dict().items():
                    values[f"{arm}/{name}/{key}"] = parameter.detach().numpy()
    np.savez(filename, **values)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("version", choices=("old", "new"))
    p.add_argument("output", type=Path)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    capture(args.checkout.resolve(), args.version, args.output)
    if args.compare:
        with np.load(args.compare) as old, np.load(args.output) as new:
            assert set(old.files) == set(new.files)
            for k in old:
                np.testing.assert_array_equal(new[k], old[k], err_msg=k)
            print(
                f"First Adam update matched exactly for all {len(old.files)} parameter/buffer arrays."
            )
