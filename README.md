# Decision-focused learning for six hybrid power transformers

Can a five-second solar forecast improve voltage control when the forecast model learns from the actions that six hybrid power transformers (HPTs) actually issue? This repository releases the three-arm, three-seed experiment on a modified unbalanced IEEE 123-bus feeder. It includes the processed trajectories, the feeder used for replay, model initializations, nine validation-selected checkpoints and all 774 test-window records.

**Main finding.** On 30 September dates, joint decision-focused learning (DFL) saved 0.313 kWh and reduced voltage violations by 100.3 seconds per 90-minute window relative to continuing mean squared error (MSE) training. Compared with freezing the forecaster, joint DFL used 0.103 kWh more per window. These comparisons describe the selected operating points. Energy savings at matched voltage risk remain an open question.

## Reproduce the results

Use Python 3.11 or 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py verify
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. Verification only needs NumPy and pandas. It checks the published artifact hashes, verifies the validation-based checkpoint choices, matches each test record to its expected window and recomputes the ten summary and paired-difference tables in `work/tables/`. The reference tables remain available in the [published run](hpt_project/runs/ieee123_expanded_cuda_v1/).

**Replay a selected model.** Install PyTorch in the same environment. The original training used PyTorch 2.3.1 with CUDA 12.1. For CPU replay:

```sh
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
python run.py replay --arm joint --seed 11 --split test_may --windows 1
```

This issues 1,080 five-second decisions and replays 5,400 seconds of operating conditions. Omit `--windows` for the entire split. Choices are `frozen`, `mse`, `joint`, seeds `11`, `23`, `37`, and splits `test_may`, `test_september`. Timings are measured on the machine running the replay. The plant holds the last issued command when a proposal fails its check or misses the deadline, so timing can also affect physical outcomes.

**Retrain all nine models.** On Linux with CUDA:

```sh
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python run.py train --gpus 0 1 2
```

Pass the CUDA device indices visible to your process. The launcher assigns devices by seed and runs the three arms for each seed concurrently. One GPU can be specified if it has sufficient memory. This command writes to `work/retrain/` and summarizes newly generated results under `work/retrained_tables/`. Starting from an empty work directory reproduces training from the supplied initialization. Reusing the directory resumes completed epoch checkpoints. The reference run took approximately 25 h 16 min including validation and testing on three RTX 3090s, with AC feasibility checks and timed deployment on CPU.

## What is compared?

Every five seconds the forecaster takes eleven seconds of observations from six PV sites and predicts the next five seconds for each site. The action network combines these forecasts, observed feeder and load conditions and device state. It issues 54 continuous commands: in-phase series voltage, quadrature series voltage and shunt reactive power for each phase of six HPTs. The online AC check and issued-action history are shared across the three arms.

| Arm | Action network training | Forecast network training |
| --- | --- | --- |
| Frozen | Physical dispatch feedback | Fixed pretrained forecaster |
| Continue MSE | Same physical feedback | Supervised PV error |
| Joint DFL | Same physical feedback | Supervised PV error and dispatch feedback through the action network |

The training code estimates execution-cost derivatives from paired action perturbations and adds the derivative of the proposal’s device and voltage penalties. In the MSE arm the forecast is detached at the actor input. The joint arm allows that action gradient to reach the forecaster. Validation selects the lowest date-balanced score including initialization. The three arms share the training budget for each seed.

## Data and outcomes

One epoch visits 78,694 five-second periods across 35 training dates: 5,254 legacy periods and 73,440 periods from 68 new 90-minute trajectories. Validation covers four dates. The May test uses four previously unevaluated dates. The September test uses 75 held-out windows on 30 historically used dates. Sites and loads represent a simulated feeder constructed from Oahu irradiance and OEDI load data. Dates are paired before averaging across days; windows from one day share a date weight.

| Split (90-minute windows) | Arm | Energy loss (kWh) | Voltage violation (s) | HPT overload (s) |
| --- | --- | ---: | ---: | ---: |
| May | Frozen | 69.626 | 804.6 | 647.8 |
| May | Continue MSE | 70.140 | 718.9 | 619.2 |
| May | Joint DFL | 69.931 | 677.1 | 616.9 |
| September | Frozen | 74.565 | 475.9 | 207.6 |
| September | Continue MSE | 74.980 | 538.5 | 208.1 |
| September | Joint DFL | 74.668 | 438.2 | 189.3 |

The September date-block 95% intervals for joint minus MSE are −0.386 to −0.245 kWh and −157.1 to −43.6 voltage-violation seconds. These intervals condition on the three trained seeds. All three reduced voltage-violation time; two of three reduced energy loss. The interval for overload duration includes zero. [Detailed results and limitations (Chinese)](hpt_project/runs/ieee123_expanded_cuda_v1/FINAL_RESULTS.md)

## Repository guide

- [Training loop](hpt_project/tools/ieee123_expanded_train.py) and [trajectory loader](hpt_project/tools/ieee123_expanded_data.py)
- [HPT steady-state model](hpt_project/tools/ieee123_six.py), [service-area configuration](hpt_project/configs/ieee123_service_areas.json), [runtime action checks](hpt_project/tools/ieee123_direct_confirm.py)
- [Execution-cost derivative](hpt_project/tools/ieee123_full_coverage.py) and [proposal-safety derivative](hpt_project/tools/ieee123_proposal_train.py)
- [Data split and provenance](hpt_project/runs/ieee123_expanded_trajectories_v1/REPORT.md), [locked manifest](hpt_project/runs/ieee123_expanded_trajectories_v1/manifest.csv) and [data attribution](DATA_SOURCES.md)

The reproducible entry points above start from the published feeder snapshot, processed data and initial weights. Some supporting modules also contain historical preparation routines that used earlier experiment directories. The release covers the expanded training and test comparison.

## License

Original project software is available under the [MIT License](LICENSE). Upstream data and the derived feeder retain their respective terms; see [data sources and notices](DATA_SOURCES.md).
