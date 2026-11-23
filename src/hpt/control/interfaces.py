"""Adapters consumed by the timed closed loop. Forecasts use native site units."""

from typing import Any, Protocol

import numpy as np


class Controller(Protocol):
    def propose(
        self,
        frame: Any,
        previous: np.ndarray,
        history: np.ndarray,
        forecast: np.ndarray,
    ) -> np.ndarray:
        """Return one normalized action for this feeder using causal information."""
        ...


class Feeder(Protocol):
    action_dim: int

    def feasible(self, frame, command, history, reference_forecast) -> bool: ...
    def score(self, frame, previous, issued, delay_seconds, realized_pv) -> dict: ...
