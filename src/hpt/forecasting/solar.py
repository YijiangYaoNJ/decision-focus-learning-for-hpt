"""Six-site residual five-second PV forecast and its supervised initialization."""

import torch
from torch import nn

from hpt.config import CHECKPOINTS


def tensor(a):
    return torch.as_tensor(a, dtype=torch.float32)


class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(66, 64), nn.SiLU(), nn.Linear(64, 30))

    def forward(self, h):
        latest = h[:, -1:]
        x = torch.cat((latest, 5 * (h[:, 1:] - h[:, :-1])), 1)
        return (latest + 0.35 * self.net(x.flatten(1)).tanh().reshape(-1, 5, 6)).clamp(
            0, 1
        )


def forecast_net(path=CHECKPOINTS / "initial/init_11.pt"):
    net = Forecaster()
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["fore"])
    return net.eval()
