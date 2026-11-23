"""One proposal, one common capability check, timed issue or hold."""

from time import perf_counter

import numpy as np

from hpt.config import DIM, SPEC
from hpt.control.interfaces import Controller, Feeder
from hpt.forecasting.solar import tensor
from hpt.physics.metrics import phase_power, scores
from hpt.physics.safety import accepted_commands, disk


class NeuralController:
    def __init__(self, actor):
        self.actor = actor

    def propose(self, frame, previous, history, forecast):
        return disk(
            self.actor.suggest(frame, previous, history, tensor(forecast[None])).numpy()
        )[0]


class IEEE123Feeder:
    action_dim = DIM

    def feasible(self, frame, command, history, reference_forecast):
        weather = phase_power(np.vstack((history[-1], reference_forecast)))
        return accepted_commands(frame, command[None], weather)[0]

    def score(self, frame, previous, issued, delay_seconds, realized_pv):
        return scores(
            frame.m,
            frame,
            previous,
            issued,
            delay_seconds,
            phase_power(realized_pv),
            "direct",
        )


def issue(
    controller: Controller, feeder: Feeder, predict, reference, frame, previous, history
):
    begin = perf_counter()
    proposal = None
    try:
        reference_path = reference(history)
        path = predict(history)
        proposal = controller.propose(frame, previous, history, path)
        if proposal.shape != (feeder.action_dim,):
            raise ValueError(f"Expected {feeder.action_dim} control values")
        status = (
            "issued"
            if feeder.feasible(frame, proposal, history, reference_path)
            else "rejected"
        )
    except ArithmeticError:
        status = "solver_error"
    delay = perf_counter() - begin
    if delay >= SPEC["coordination_seconds"]:
        status = "timeout"
    return proposal if status == "issued" else None, status, delay, proposal
