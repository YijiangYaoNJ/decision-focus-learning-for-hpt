"""A small deterministic test for adding a conventional controller or another feeder."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hpt.evaluation.replay import replay_window


class TwoBusController:
    def propose(self, frame, previous, history, forecast):
        return np.array([previous[0] + forecast[0]])


class TestControllerInterface(unittest.TestCase):
    def test_rejected_action_is_held_and_both_controllers_receive_causal_input(self):
        class Window:
            initial = np.array([0.0])
            d = {"past": [[1.0], [2.0], [3.0]], "truth": [[5.0], [6.0], [7.0]]}

            def plant(self, i):
                return i

        class Feeder:
            action_dim = 1

            def feasible(self, frame, command, history, reference_forecast):
                self.assert_reference(reference_forecast, [-1])
                return frame != 1

            def score(self, frame, previous, issued, delay_seconds, realized_pv):
                active = previous if issued is None else issued
                return dict(
                    kwh=active[0] * realized_pv[0],
                    voltage_s=0,
                    overload_s=0,
                    v_peak=0,
                    overload_peak=0,
                )

        feeder = Feeder()
        feeder.assert_reference = np.testing.assert_array_equal
        output = replay_window(
            Window(), TwoBusController(), feeder, lambda h: h, lambda h: [-1.0]
        )
        self.assertEqual(output["issued"], 2)
        self.assertEqual(output["kwh"], 1 * 5 + 1 * 6 + 4 * 7)
