# Analysis plan for the CUDA replication of the G2 primary comparison

Written and committed before any CUDA run of `configs/full_cuda.yaml`, `full_cuda_more.yaml` or `full_cuda_sweep.yaml`. The commit that adds this file is the record of what was decided in advance; departures are reported as departures. The MPS results it replicates are in [`SCIENTIFIC_AUDIT.md`](SCIENTIFIC_AUDIT.md), F15 ([`FULL_BUDGET_PLAN.md`](FULL_BUDGET_PLAN.md) for their plan).

## Question

Does the main MPS result, that training with the corrected (`consistent`) residual at physics weight 0.1 lowers the test error of the FNO by about a fifth compared with data-only training (MPS, five seeds: -21.6 % [-23.5, -19.6]), hold when the same configurations are trained on an NVIDIA GPU?

CUDA and MPS results are not expected to be numerically identical: the floating-point kernels differ, and on CUDA the finite-difference residual runs on the GPU (NVIDIA Warp), whereas on MPS it ran on the CPU. The replication concerns the effect, not the individual numbers.

## Studies

Every training setting is that of `configs/full.yaml` and `configs/full_mps.yaml` (5100 optimiser steps: 50 epochs of 102 samples, batch size 1, Adam 1e-3 with the upstream per-step decay, the upstream FNO); only the device differs from the MPS studies. `tests/test_config.py` checks this.

| Config | Mirrors (MPS) | Runs | Content |
|---|---|---|---|
| `full_cuda` | `full_mps` | 9 | data only, `shipped` 0.1, `consistent` 0.1; seeds 0, 1, 2 |
| `full_cuda_more` | `full_mps_more` | 6 | the same settings; seeds 3, 4 |
| `full_cuda_sweep` | `full_mps_sweep` | 18 | both residuals at weights 0.001, 0.01, 1; seeds 0, 1, 2 |

- **Minimal replication** (`PLAN = "minimal"` in `kaggle/run_cuda.ipynb`): `full_cuda` + `full_cuda_more`, 15 runs. This is all the primary criterion needs.
- **G2 replication** (`PLAN = "g2"`): `full_cuda` + `full_cuda_sweep`, 27 runs, the same runs as `configs/full.yaml`.
- **Everything** (`PLAN = "all"`): 33 runs.

## Primary effect and interval

For seed $s \in \{0, 1, 2, 3, 4\}$ let $E_s(x)$ be the mean over the 256 test samples of the per-sample relative $L^2$ error of the model trained with setting $x$ on CUDA (`test_rel_l2.mean` in `eval_metrics.json`, recomputed from the checkpoint by `scripts/evaluate_solver_residual.py` and required to agree to 1e-5 relative).

- Paired difference: $d_s = E_s(\texttt{consistent}\ 0.1) - E_s(\text{data only})$.
- Mean and interval: $\bar d$ and $\bar d \pm t_{0.975,\,4}\, \mathrm{sd}(d_s)/\sqrt 5$, with the sample standard deviation over the five seeds and $t_{0.975,4} = 2.776$.
- Relative effect: $\bar d / \bar E(\text{data only})$, and the interval divided by the same $\bar E(\text{data only})$, the mean data-only error over the five CUDA seeds. The control mean is treated as fixed, as in the MPS analysis.

## Primary criterion

**The result replicates if the upper endpoint of the relative 95 % interval is below -5 %**, that is, if the improvement is larger than 5 % of the data-only error with 95 % confidence over seeds.

- If it is met: "replicated on CUDA".
- If the interval lies entirely below zero but its upper endpoint is at or above -5 %: "not replicated by the criterion; the direction agrees".
- Otherwise: "not replicated".
- If fewer than five seeds of either setting are complete, the criterion is "not evaluable" and is not evaluated on fewer seeds.

The result is reported whatever it is. No seeds are added after the five, and the criterion is not changed after any CUDA run.

## Secondary, descriptive

- The same analysis for seeds 0 to 2 only (as reported first on MPS), and for `shipped` 0.1 minus data only, with the reading rule of `FULL_BUDGET_PLAN.md` (improvement, degradation, no material effect within ±5 % of the control mean, inconclusive).
- Error near permeability jumps and on constant-`Kcoeff` cells; exact (generator-operator) residual and cell-grid residual as geometric-mean ratios to data only.
- Device differences, paired by seed (the seed sets the initial weights and the sample order on both devices): $E^{\text{CUDA}}_s(x) - E^{\text{MPS}}_s(x)$ per setting, and the difference of the paired effects $d^{\text{CUDA}}_s - d^{\text{MPS}}_s$, with 95 % t intervals over seeds. These describe how far the devices differ; no equivalence claim is made from them.
- If `full_cuda_sweep` exists: every sweep setting minus data only, seeds 0 to 2, with the MPS value beside it.
- Per-seed values of everything above.

## Execution and provenance

- `kaggle/run_cuda.ipynb` checks out one commit by its full SHA, verifies a clean tree, runs the data check and the tests, a CUDA smoke study and a one-epoch timing probe, and starts training only if `PLAN` is set and the projected time fits `SESSION_BUDGET_HOURS`. It never reduces a configuration.
- Every study is trained with `--resume`: finished runs are skipped, the study metadata is written once and a resume with a different configuration is refused (`pino_darcy.study.check_resumable`).
- The audit evaluations run after all training of the session, so that their output files do not mark later studies as dirty.
- The results are published to `results/full_cuda*/` and analysed by `scripts/cuda_replication.py`, which writes `results/audit/cuda_replication.json` and `figures/cuda_replication.png`. `scripts/results.py publish` refuses to overwrite an existing results directory, and no CUDA study name is that of an MPS study (`tests/test_config.py`), so the MPS records cannot be changed by the replication.

## Outcome (added after the runs, 2026-11-08)

Everything above this section was committed before any CUDA run and has not been changed. The minimal replication (`PLAN = "minimal"`, 15 runs) was executed once; `full_cuda_sweep` was not run.

**Execution.** Kaggle notebook session, commit `416bfe6`, clean checkout (`git_dirty: false` in both `study_metadata.json`), one NVIDIA Tesla T4 (driver 580.178.04), CUDA 13.2, PyTorch 2.15.0+cu132, Python 3.12.3 in an isolated environment, PhysicsNeMo 2.3.0a0 at the pinned upstream commit `b45a5c8`, Warp 1.19.0. Before training the session verified the dataset against the tracked checksums, passed the 116 tests of that commit and a CUDA smoke study. Training took 190 to 194 s per run (23 to 24 ms per optimiser step, peak 1.0 GB of GPU memory), 0.80 h for the 15 runs and 0.98 h for the whole session. All 15 runs have 5100 optimiser steps, a 50-epoch history, test metrics and a final checkpoint.

**Recomputation.** The test error recomputed from every checkpoint in the session agrees with the stored value to 9.8e-8 relative (requirement: 1e-5); recomputed again locally on Apple MPS from the downloaded checkpoints, to 9.1e-6. `scripts/cuda_replication.py` gives the same output in the session and locally, and `tests/test_cuda_replication.py` recomputes the primary interval from the per-run records with the formula above, independently of that script.

**Primary result.** `consistent` 0.1 minus data only, seeds 0 to 4: mean data-only error 0.03827, `consistent` 0.1 0.03003; relative change **-21.5 % [-23.5, -19.6]**, lower in all five seeds. The upper endpoint -19.6 % is below -5 %: **replicated on CUDA**.

**Secondary results** (descriptive, seeds 0 to 4 unless stated; [`results/audit/cuda_replication.json`](../results/audit/cuda_replication.json), [`figures/cuda_replication.png`](../figures/cuda_replication.png)):

| Comparison | CUDA | MPS (same seeds) |
|---|---|---|
| `consistent` 0.1 - data only, seeds 0 to 2 | -21.7 % [-27.1, -16.2] | -21.7 % [-27.1, -16.3] |
| `consistent` 0.1 - data only, seeds 0 to 4 | -21.5 % [-23.5, -19.6] | -21.6 % [-23.5, -19.6] |
| `shipped` 0.1 - data only (reading) | -1.1 % [-3.8, +1.6], no material effect | -1.1 % [-3.8, +1.5], no material effect |
| `consistent` 0.1, near jumps / constant-`Kcoeff` cells | -16.1 % [-18.2, -14.1] / -24.2 % [-26.6, -21.8] | |
| `consistent` 0.1, exact residual (geometric-mean ratio to data only) | 0.20 [0.18, 0.22] | |
| `consistent` 0.1, training-split error | +121 % [+113, +129] | |

**Device differences.** Paired by seed, CUDA minus MPS: data only differs by 8e-7 of the MPS mean error, `consistent` 0.1 by 2e-5, `shipped` 0.1 by 2e-4; per run, data-only test errors agree to at most 3e-6 relative and the physics-informed ones to at most 1.1e-3. The difference of the paired effects (`consistent` 0.1 minus data only) is +7e-7 [-1.1e-5, +1.3e-5] in absolute error. The devices are not bitwise equal: every training history differs in its per-epoch data loss (by up to 1e-5 relative for data only and 5e-3 for the physics-informed runs). The closeness is larger than the plan anticipated; it is reported as an observation, and no equivalence claim is made from it, as stated above.

**Departures from the plan.** None in the analysis. Package versions differ from the MPS runs because `requirements.txt` is not pinned exactly (PyTorch 2.15.0 against 2.14.1, NumPy 2.5.4 against 2.5.3); they are recorded in the metadata. The local analysis file records `git_dirty: true` because the CUDA results were untracked when it was written; `git diff HEAD` was empty.

**Archives.** The session's archives are kept outside Git in `runs/kaggle/pino-darcy-cuda-minimal-416bfe6/` with a `SHA256SUMS` manifest; every file in them matches the extracted records in `results/full_cuda*` and checkpoints in `runs/full_cuda*`.
