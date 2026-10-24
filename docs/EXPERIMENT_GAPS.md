# Experiment gaps

Experiments and controls that the current evidence does not cover, in order of scientific value per unit of compute. None of them has been run. Every command below is exact and runs from the repository root with the project environment (`.venv`, `requirements.lock`). Runtime estimates are projections from the measured cost of the reduced study on an Apple M3 Pro (171 ms per optimiser step without the PDE term, 217 to 238 ms with it; 102 steps per epoch) and are not measurements.

| ID | Gap | Why it matters | Cost | Status |
|---|---|---|---|---|
| G1 | Training seeds | All conclusions rest on seed 0. The accuracy differences between data-only and `consistent` at weights up to 0.1 (at most 0.0005) are far below any plausible seed-to-seed spread, which is unmeasured. | 6 runs, about 45 min MPS | PENDING |
| G2 | Upstream training budget | Validation error is still falling at epoch 15 in 8 of 9 runs. Rankings of non-converged models may change. | 27 runs of 5100 steps; about 8.4 h MPS (too long locally); CUDA time unknown, projected by the notebook from a one-epoch probe before anything starts | PENDING (CUDA) |
| G3 | Residual coefficient `coeff` instead of `Kcoeff` | The residual uses the smoothed model input, not the coefficient the data were generated with (audit F4). The bias it introduces (1.7 % mean relative L2 between the two solutions) is of the order of the effects being measured. | 4 runs, about 30 min MPS, after a code change | NOT IMPLEMENTED |
| G4 | Training with the solver-consistent residual | `pino_darcy.reference.solver_residual` is zero for the reference everywhere, including at jumps. Training with it would separate "physics helps" from "the discrete operator is wrong at jumps". Needs a differentiable torch port and a treatment of the cropped edge (extension plus margin, validated in `results/audit/reference_check.json`). | implementation plus about 8 runs | NOT IMPLEMENTED |
| G5 | Residual on unlabelled inputs | Here the residual is evaluated only on the 102 labelled training samples, where the data term already constrains the output. The regime in which PINO-type losses are expected to help (extra permeability fields without pressure, or fewer labels) is not tested. | design decision; at least 10 runs | NOT DESIGNED |
| G6 | Training-set size | Physics weighting at 102 samples only. A sweep over 25, 50, 102 samples would show whether the effect depends on data scarcity. | 3 sizes x 3 settings x 3 seeds | NOT DESIGNED |
| G7 | DeepONet variant with automatic differentiation | Out of scope; listed for completeness. | not estimated | OUT OF SCOPE |

## G1: training seeds (recommended next step, local)

```bash
python scripts/train.py --config local name=local_seeds "study.seeds=[1,2]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]"
```

Six runs (data-only, `shipped` 0.1, `consistent` 0.1 for seeds 1 and 2), written to `runs/local_seeds/`. Before starting: check `memory_pressure` and that no other training job is running on the machine (the reduced study shared the machine and its timings vary for that reason). Expected scientific value: the first estimate of seed-to-seed spread, which decides whether the 0.0480 against 0.0475 difference, and the 9 % degradation of `shipped` at the upstream default, are reproducible. To compare with seed 0, run `python scripts/evaluate_solver_residual.py --config local name=local_seeds "study.seeds=[1,2]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]"` as well.

## G2: upstream budget on CUDA

Use `kaggle/run_cuda.ipynb`:

1. Push the commit to run and set `REF` to its full 40-character SHA.
2. Accelerator: one NVIDIA GPU. Internet on (0.8 GB dataset download).
3. Run once with `RUN_FULL = False` to obtain the timing projection; set `RUN_FULL = True` only if it fits `SESSION_BUDGET_HOURS`. The notebook never reduces the configuration.
4. Integrate as described in the README section "Study at the upstream budget on CUDA", then `python scripts/evaluate.py --config full --verify device=mps` and `python scripts/evaluate_solver_residual.py --config full device=mps` locally.

Colab: the same cells run after replacing `/kaggle/working` by `/content` and removing the Kaggle secret lookup; the dataset download needs Google Drive access from the session.

Peak memory is unmeasured on CUDA; the FNO at 240 x 240 with batch size 1 is small (2.4 M parameters), so any current data-centre or Kaggle GPU should be sufficient. This is an expectation, not a measurement.

## Missing controls inside the existing design

- **Paired statistics.** The README reports the per-sample standard deviation; paired differences over the shared test samples are more informative and are given in the audit (F6). They still measure test-sample variability only.
- **Region-resolved error.** The summary reports residuals by region but not the error by region. `results/audit/local_solver_residual.json` adds it for the reduced study.
- **Tuning parity.** Learning rate and schedule are the upstream values for all weights. A physics weight that changes the loss scale by a factor of 2 to 3 (weight 1) may prefer a different learning rate; the comparison is at fixed hyperparameters, which should be stated as such.
- **Bitwise reproducibility on MPS.** Seeds are set, but MPS kernels are not guaranteed deterministic; re-running a run and comparing histories has not been done.
