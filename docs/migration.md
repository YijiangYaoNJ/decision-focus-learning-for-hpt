# Engineering migration from v0.1.0

The baseline is the annotated tag `v0.1.0` at `6f520a38cb3fecb0e67aaabbea1228833f748274`. The comparison runs the tagged implementation in its own checkout and the engineering implementation in a separate Python process. Both use the same CPU, single-thread settings, model weights, observations, previous actions and perturbation seeds.

| Comparison | Coverage | Evidence |
| --- | --- | --- |
| Input and reference integrity | All 132 previously checksummed files | Each moved file has exactly its original SHA-256. The new manifest also pins the archived experiment configuration. |
| Forecast, proposed action and execution | Four recorded training states for each of three arms and three selected seeds (36 cases) | Compare all 30 forecast entries, all 54 command entries, the executed command and the physical objective. One chosen state activates the proposal safety penalty. |
| Gradients | The same 36 cases | Compare the action-space execution derivative, proposal-safety derivative and parameter gradients of both networks. Zero gradients for the frozen forecaster are checked alongside nonzero cases. |
| First training step | All three arms with seed 11 | Run the real paired data loader and optimizer through the first Adam update. Compare the 51 model parameter and buffer arrays exactly. |
| Closed-loop replay | First May window, 2010-05-24, joint DFL, seed 11 | Compare the proposed and issued vectors (including holds) and physical outcomes at every one of 1,080 five-second decisions. Both runs issued 948 commands, consumed 50.80741241886356 kWh in losses and recorded zero voltage-violation or overload seconds. Wall-clock latency is measured anew. |
| Published study tables | 86 held-out windows × three arms × three seeds | Rebuild ten tables from the 774 original window records and compare each to the published artifact. |

See the [recorded numerical differences](migration-parity.json). The checkpoint replay regenerates one full trajectory. The 774-record check re-aggregates the archived tests. Training equivalence was exercised through gradients and one optimizer update; a repeat of the full multi-GPU training job was outside this migration.

## Run the comparison

From the repository root on a machine with the replay dependencies installed:

```sh
git worktree add --detach work/baseline-v010 v0.1.0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python tests/capture_v010.py work/baseline-v010 work/oracle
python tests/compare_v010.py work/oracle runs/migration
python tests/first_update.py work/baseline-v010 old work/old_update.npz
python tests/first_update.py . new work/new_update.npz --compare work/old_update.npz
python -m unittest discover -s tests -p 'test_*.py'
python run.py verify
```

The regression probes read trusted checkpoints from the named checkout. The interface test substitutes a one-control feeder and deterministic controller and checks that a rejected proposal carries the last issued action into the next cycle.
