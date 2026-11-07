# Experiment gaps

Experiments and controls that the current evidence does not cover, in order of scientific value per unit of compute. G1 was run on 2026-10-24; G2 to G6 were run on Apple MPS on 2026-11-06 and 2026-11-07 (section "G2 to G6: executed on MPS" below); a CUDA replication of G2 and G7 have not been run. Every command below is exact and runs from the repository root with the project environment (`.venv`, `requirements.lock`). Runtime estimates are projections from the measured cost of the reduced study on an Apple M3 Pro (171 ms per optimiser step without the PDE term, 217 to 238 ms with it; 102 steps per epoch) and are not measurements.

| ID | Gap | Why it matters | Cost | Status |
|---|---|---|---|---|
| G1 | Training seeds | All conclusions rested on seed 0. The accuracy differences between data-only and `consistent` at weights up to 0.1 (at most 0.0005) were below the unmeasured seed-to-seed spread. | 6 runs; measured 17.7 min training on MPS | DONE: `SCIENTIFIC_AUDIT.md` F14 |
| G2 | Upstream training budget | Validation error is still falling at epoch 15 in 8 of 9 runs. Rankings of non-converged models may change. | 27 runs of 5100 steps; about 8.4 h MPS (too long locally); CUDA time unknown, projected by the notebook from a one-epoch probe before anything starts | DONE on MPS (`SCIENTIFIC_AUDIT.md` F15); CUDA replication not run |
| G3 | Residual coefficient `coeff` instead of `Kcoeff` | The residual uses the smoothed model input, not the coefficient the data were generated with (audit F4). The bias it introduces (1.7 % mean relative L2 between the two solutions) is of the order of the effects being measured. | implemented (`beaac7f`); 3 runs at the G2 budget | DONE on MPS: F16 |
| G4 | Training with the solver-consistent residual | `pino_darcy.reference.solver_residual` is zero for the reference everywhere, including at jumps. Training with it would separate "physics helps" from "the discrete operator is wrong at jumps". Needs a differentiable torch port and a treatment of the cropped edge (extension plus margin, validated in `results/audit/reference_check.json`). | implemented (`beaac7f`); 9 runs (incl. `solver_kcoeff`, weight 1) | DONE on MPS: F16 |
| G5 | Residual on unlabelled inputs | Here the residual is evaluated only on the 102 labelled training samples, where the data term already constrains the output. The regime in which PINO-type losses are expected to help (extra permeability fields without pressure, or fewer labels) is not tested. | designed in `OPERATOR_AND_LABELS_PLAN.md`; 6 runs | DONE on MPS: F17 |
| G6 | Training-set size | Physics weighting at 102 samples only. A sweep over 25, 50, 102 samples would show whether the effect depends on data scarcity. | 2 new sizes x 2 settings x 3 seeds (102 from G2/G4) | DONE on MPS: F17 |
| G7 | DeepONet variant with automatic differentiation | Out of scope; listed for completeness. | not estimated | OUT OF SCOPE |

## G1: training seeds (done)

```bash
python scripts/train.py --config local name=local_seeds "study.seeds=[1,2]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]"
```

Six runs (data-only, `shipped` 0.1, `consistent` 0.1 for seeds 1 and 2), written to `runs/local_seeds/`. Before starting: check `memory_pressure` and that no other training job is running on the machine (the reduced study shared the machine and its timings vary for that reason). Expected scientific value: the first estimate of seed-to-seed spread, which decides whether the 0.0480 against 0.0475 difference, and the 9 % degradation of `shipped` at the upstream default, are reproducible. To compare with seed 0, run `python scripts/evaluate_solver_residual.py --config local name=local_seeds "study.seeds=[1,2]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]"` as well.

**Executed 2026-10-24** at commit `4c18b90`, with `evaluation.plot_samples=3` added (evaluation output only; the seed-0 value). Records in `results/local_seeds/` (published with `scripts/results.py publish runs/local_seeds`), analysis by `scripts/seed_study.py` in `results/audit/seed_study.json` and `figures/seed_study.png`. Outcome over seeds 0, 1, 2:

- Seed-to-seed standard deviation of the data-only error: 0.0013 (2.7 %).
- `consistent` 0.1 minus data-only: -0.0004, 95 % t interval over seeds [-0.0070, +0.0061]; lower in two seeds, higher in one. The 0.0480 against 0.0475 difference of seed 0 is not reproducible as an effect.
- `shipped` 0.1 minus data-only: +0.0040 (+8 %), [-0.0002, +0.0081]; worse in all three seeds. The 9 % degradation reproduces in direction and approximate size; the interval just includes zero.
- The residual reduction reproduces in every seed under both residuals; the error near jumps is higher in every seed for both weighted settings.

Remaining gap: with three seeds an effect of `consistent` 0.1 between -14 % and +13 % cannot be excluded. Each further seed costs three runs of about 3 minutes each on an unloaded M3 Pro:

```bash
python scripts/train.py --config local name=local_seeds_more "study.seeds=[3,4,5,6,7]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]" evaluation.plot_samples=3
```

(not run; `scripts/seed_study.py` reads seeds 0 to 2 and would need its `SEEDS` and `STUDIES` extended).

## G2 to G6: executed on MPS

Pre-registered in [`FULL_BUDGET_PLAN.md`](FULL_BUDGET_PLAN.md) (G2, commit `eaca64c`) and [`OPERATOR_AND_LABELS_PLAN.md`](OPERATOR_AND_LABELS_PLAN.md) (G3 to G6, commit `beaac7f`), each committed before its runs. All runs: 5100 optimiser steps (the upstream budget), seeds 0 to 2, Apple M3 Pro MPS, each study trained from a git worktree at the plan's commit. A machine-wide GPU lock (`~/.cache/physicsnemo/gpu.lock`) serialised them with other projects' jobs. Commands, from the repository root:

```bash
python scripts/train.py --config full_mps                 # G2 stage 1, 9 runs
python scripts/train.py --config full_mps_sweep           # G2 stage 2, 18 runs
python scripts/train.py --config full_mps name=full_mps_more "study.seeds=[3,4]"   # G2 stage 3, 6 runs
python scripts/train.py --config full_mps_operator        # G3, G4, 9 runs
python scripts/train.py --config full_mps_solver_w1       # G4 at weight 1, 3 runs
python scripts/train.py --config labels25                 # G6, 6 runs; likewise labels50
python scripts/train.py --config labels25_unlabelled      # G5, 3 runs; likewise labels50_unlabelled
# after each study (same --config and overrides):
python scripts/evaluate_solver_residual.py --config <config>
python scripts/evaluate_generalisation.py --config <config>
python scripts/results.py publish runs/<study>
python scripts/full_study.py                              # all statistics and figures/full_*.png
```

Outcome (95 % t intervals over seeds, paired by seed; details and caveats in the audit):

- G2 (F15): `consistent` 0.1 minus data only -21.7 % [-27.1, -16.3], every seed (primary). Weight 1 inconclusive; `shipped` 1 +142 %. The near-jump degradation seen at 15 epochs does not occur. Stage 3 (seeds 3 and 4) was triggered by the pre-registered stopping rule; with n = 5, -21.6 % [-23.5, -19.6], lower in all five seeds; `shipped` 0.1 (the upstream default) has no material effect (-1.1 % [-3.8, +1.5]).
- G3, G4 (F16): `solver` 0.1 minus data only -19.4 % [-26.8, -12.0] (primary). All four residual variants improve by 18 to 22 %; the exact operator is not better than the cell-grid one; `coeff` is slightly worse than `Kcoeff`; `solver` 1 inconclusive.
- G5, G6 (F17): with 25 and 50 labels, `solver` 0.1 improves by 27 % and 30 %, and with the unlabelled fields by 42 % and 39 % (four primary comparisons, Holm, every seed). The unlabelled runs use about twice the compute per step.

Remaining gaps after G2 to G6: the CUDA replication below; tuning parity (all settings use the upstream learning rate and schedule); controls with other regularisers (early stopping, weight decay), which would test the regularisation reading of F15; and other datasets and architectures.

## G2 replication on CUDA (not run)

The plan for G2 targeted CUDA; G2 was run on MPS instead. The replication is prepared (2026-11-08) and pre-registered in [`CUDA_REPLICATION_PLAN.md`](CUDA_REPLICATION_PLAN.md): configurations `full_cuda` (9 runs), `full_cuda_more` (6) and `full_cuda_sweep` (18), mirroring `full_mps`, `full_mps_more` and `full_mps_sweep`; criterion: the upper end of the five-seed relative 95 % interval of `consistent` 0.1 minus data only below -5 %. Status: NOT RUN.

Use `kaggle/run_cuda.ipynb`:

1. Push the commit to run and set `REF` to its full 40-character SHA.
2. Accelerator: one NVIDIA GPU. Internet on (0.8 GB dataset download).
3. Run once with `PLAN = None` to check the session and obtain the timing projection; then set `PLAN = "minimal"` (15 runs), `"g2"` (27) or `"all"` (33). Training starts only if the projection fits `SESSION_BUDGET_HOURS`; the notebook never reduces a configuration. A session that cannot finish the plan continues in the next one from its archives (`RESTORE_DIR`).
4. Integrate as described in the README section "Replication on CUDA"; `scripts/cuda_replication.py` evaluates the criterion and compares with the MPS results.

Colab: the same cells run after replacing `/kaggle/working` by `/content` and removing the Kaggle secret lookup; the dataset download needs Google Drive access from the session.

Peak memory is unmeasured on CUDA; the FNO at 240 x 240 with batch size 1 is small (2.4 M parameters), so any current data-centre or Kaggle GPU should be sufficient. This is an expectation, not a measurement. Storage, from the MPS runs: about 28 MB of checkpoints per run (0.4 GB for the minimal plan, 0.9 GB for all 33 runs), the same again for the eval bundles in the session, plus the 0.8 GB dataset archive and its 0.5 GB of split files; the tracked records add about 1.5 to 3 MB per study.

## Missing controls inside the existing design

- **Paired statistics.** The README reports the per-sample standard deviation; paired differences over the shared test samples are more informative and are given in the audit (F6). They still measure test-sample variability only.
- **Region-resolved error.** The summary reports residuals by region but not the error by region. `results/audit/local_solver_residual.json` adds it for the reduced study; the 50-epoch studies report it in `results/audit/full_study.json` (done).
- **Tuning parity.** Learning rate and schedule are the upstream values for all weights. A physics weight that changes the loss scale by a factor of 2 to 3 (weight 1) may prefer a different learning rate; the comparison is at fixed hyperparameters, which should be stated as such.
- **Seed variability at other weights.** At 15 epochs only weight 0.1 has more than one seed; at 50 epochs every weight has three seeds (`full_mps_sweep`, done).
- **Bitwise reproducibility on MPS.** Seeds are set, but MPS kernels are not guaranteed deterministic. Checked on 2026-11-06 (`results/audit/determinism.json`): for data only and `consistent` 0.1, two one-epoch runs with the same seed gave identical parameters on this machine and software; MPS and CPU parameters differ by up to 0.008 and 0.03 (maximum absolute difference). Not checked for the `solver` residual, for the unlabelled pool, for full-length runs, or across machines and software versions.
