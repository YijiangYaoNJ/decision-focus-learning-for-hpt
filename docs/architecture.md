# How one dispatch decision moves through the project

The IEEE123 experiment uses six service-area transformers. Each transformer receives three commands per phase: in-phase series voltage, quadrature series voltage and shunt reactive power. Five seconds later the process repeats with the command that was actually issued. Start with [the experiment configuration](../configs/ieee123.json) and [the service-area connections](../configs/ieee123_service_areas.json).

| Step | What to inspect | Contract |
| --- | --- | --- |
| 1. Load a trajectory | [data/trajectories.py](../src/hpt/data/trajectories.py) | A window supplies 1,080 chronological histories, realized PV, a feeder frame per load slot and one initial action. The manifest gives the exact dates and window offsets. |
| 2. Forecast PV | [forecasting/solar.py](../src/hpt/forecasting/solar.py) | Eleven seconds at six sites produce a five-by-six forecast of normalized PV. |
| 3. Propose an action | [control/observations.py](../src/hpt/control/observations.py) and [control/policy.py](../src/hpt/control/policy.py) | The actor uses the forecast, causal global inputs, per-load injections and local equipment measurements to propose 54 normalized commands. |
| 4. Screen and issue | [control/execution.py](../src/hpt/control/execution.py) and [physics/safety.py](../src/hpt/physics/safety.py) | All arms use the same frozen forecast for the AC capability check. The controller has five seconds to issue its proposal. The previous command remains active after rejection or timeout. |
| 5. Measure realized service | [physics/metrics.py](../src/hpt/physics/metrics.py) and [evaluation/replay.py](../src/hpt/evaluation/replay.py) | Replay the issued command with measured PV at one-second resolution. Include network and converter losses, voltage excursion and overload. Preserve the previous action until the issue delay has elapsed. |
| 6. Learn from consequences | [training/feedback.py](../src/hpt/training/feedback.py), [objectives.py](../src/hpt/training/objectives.py) and [epoch.py](../src/hpt/training/epoch.py) | The execution gradient comes from paired perturbations. The proposal gradient measures active voltage and equipment penalties. All arms update the actor; only joint DFL passes this action gradient through the forecast input. |

The validation score combines normalized energy with four safety terms. Its coefficients, training weights, optimizer rates and early-stopping budget are in the experiment JSON. The exact source archive hashes and window membership are fixed in [the dataset lock](../data/trajectories/lock.json) and [manifest](../data/trajectories/manifest.csv). Published outputs retain their [original training protocol](../results/reference/protocol.json) and [checkpoint selection](../results/reference/selection.json). The reconstruction of that configuration in the current schema is [stored alongside them](../results/reference/experiment.json).

## Adding a controller or feeder

[control/interfaces.py](../src/hpt/control/interfaces.py) defines the contracts used by [replay_window](../src/hpt/evaluation/replay.py). A conventional optimizer implements `propose(frame, previous, history, forecast)` and returns its normalized candidate. Supply that controller to the same replay loop with the IEEE123 feeder, predictor and fixed screening forecast. The optimizer receives the current feeder frame, including the network and load state. Realized future PV enters the scoring step after the decision.

A feeder adapter specifies its action dimension, checks candidates against its physical model, and scores the previous and issued actions on the realized interval. Its trajectory supplies compatible frames, histories and an initial action. The IEEE123 adapter is the reference implementation. A learned policy or training objective on a feeder with different phase connections will also need suitable device features and physical sensitivities.

## Output ownership

- `data/` contains locked trajectories, the feeder snapshot and supporting manifests.
- `checkpoints/initial/` and `checkpoints/preference/` provide initialization assets. `checkpoints/selected/<arm>_<seed>/` contains each published validation-selected network pair.
- `results/reference/` contains immutable evidence: window-level CSVs in the arm/seed folders, validation scores, final tables and the archived report.
- `runs/` holds locally generated verification tables, replays and retraining checkpoints. Resume a retrain with its original configuration; a fresh run uses an empty retraining output. All commands are invoked from the repository root.
