"""Forecast-conditioned global action network with a shared equipment head."""

import json

import numpy as np
import torch
from torch import nn

from hpt.config import CHECKPOINTS, sha
from hpt.control.observations import global_features, local_features
from hpt.forecasting.solar import Forecaster, forecast_net, tensor

PREVIOUS = CHECKPOINTS / "preference"


class Action(nn.Module):
    def __init__(self, seed):
        super().__init__()
        with np.load(PREVIOUS / f"preference_{seed}.npz") as z:
            assert (
                sha(PREVIOUS / f"preference_{seed}.npz")
                == json.loads((PREVIOUS / f"preference_{seed}.json").read_text())[
                    "weights_sha256"
                ]
            )
            self.register_buffer("mean", tensor(z["mean"]))
            self.register_buffer("scale", tensor(z["scale"]))
            self.body = nn.Sequential(
                nn.Linear(len(self.mean) + 1, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
            )
            self.action = nn.Linear(128, 54)
            self.load_state_dict({k: tensor(z[k]) for k in z.files})
        self.future = nn.Linear(30, 64, bias=False)
        nn.init.normal_(self.future.weight, std=0.025)

    def forward(self, x, trajectory, latest, risk):
        h = torch.cat(((x - self.mean) / self.scale, risk[:, None]), 1)
        # The 30 ordered site/horizon changes enter before the first activation.
        h = self.body[0](h) + self.future(
            5 * (trajectory - latest[:, None, :]).flatten(1)
        )
        h = self.body[3](self.body[2](self.body[1](h)))
        a = self.action(torch.cat((h, h * risk[:, None]), 1)).tanh().reshape(-1, 18, 3)
        uv = a[:, :, :2]
        return torch.cat(
            (uv / uv.norm(dim=2, keepdim=True).clamp_min(1), a[:, :, 2:]), 2
        ).flatten(1)


def load_models(seed, path):
    f = Forecaster()
    a = Action(seed)
    w = torch.load(path, weights_only=True)
    f.load_state_dict(w["fore"])
    a.load_state_dict(w["action"])
    return f, a


class DispatchActor(nn.Module):
    def __init__(self, seed, base=None):
        super().__init__()
        self.base = Action(seed)
        if base is not None:
            self.base.load_state_dict(base)
        self.equipment = nn.Sequential(nn.Linear(15, 16), nn.SiLU(), nn.Linear(16, 9))
        nn.init.zeros_(self.equipment[-1].weight)
        nn.init.zeros_(self.equipment[-1].bias)

    def forward(self, x, past, path, local):
        base = self.base(x, path, past[:, -1], x.new_full((len(x),), 0.5))
        a = (
            (base.reshape(-1, 6, 9) + 0.25 * self.equipment(local).tanh())
            .reshape(-1, 18, 3)
            .clamp(-1, 1)
        )
        uv = a[:, :, :2]
        return torch.cat(
            (uv / uv.norm(dim=2, keepdim=True).clamp_min(1), a[:, :, 2:]), 2
        ).flatten(1)


class DirectActor(DispatchActor):
    def __init__(self, seed, load_count):
        _, previous = load_models(seed, (CHECKPOINTS / "initial") / f"init_{seed}.pt")
        super().__init__(seed, previous.state_dict())
        first = self.base.body[0]
        extra = 2 * load_count
        expanded = nn.Linear(first.in_features + extra, first.out_features)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, : first.in_features - 1].copy_(first.weight[:, :-1])
            expanded.weight[:, -1].copy_(first.weight[:, -1])
            expanded.bias.copy_(first.bias)
        self.base.body[0] = expanded
        self.base.mean = torch.cat((self.base.mean, torch.zeros(extra)))
        self.base.scale = torch.cat((self.base.scale, torch.ones(extra)))

    def suggest(self, plant, last, history, forecast):
        x = global_features(plant, history)
        local = local_features(plant, last, history)
        return self(
            tensor(x[None]), tensor(history[None]), forecast, tensor(local[None])
        )


def networks(seed, model):
    torch.manual_seed(seed)
    actor = DirectActor(seed, len(model.lp))
    fore = forecast_net()
    return actor, fore
