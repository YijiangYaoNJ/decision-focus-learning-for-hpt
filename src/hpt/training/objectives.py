"""Arm-specific gradient routing and supervised forecast regularization."""

from hpt.config import SPEC


def policy_forecast(forecast, arm):
    if arm not in SPEC["arms"]:
        raise ValueError(arm)
    return forecast if arm == "joint" else forecast.detach()


def training_loss(actions, physical_gradient, arm, forecaster, histories, targets):
    feedback = (actions * physical_gradient).sum(1).mean()
    if arm == "frozen":
        return feedback
    mse = (forecaster(histories) - targets).square().mean()
    return feedback + SPEC["loss"]["mse_weight"] * mse
