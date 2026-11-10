# Analysis plan for the full-budget study

Written and committed before any 50-epoch run was trained. The commit that adds this file is the record of what was decided in advance. Departures from it are to be reported as departures.

## Question

At the upstream training budget (50 epochs, 5100 optimiser steps), does the data-consistent PDE residual change the test error of the FNO relative to data-only training, and does the 15-epoch picture of `SCIENTIFIC_AUDIT.md` F14 survive convergence?

## Design

Training settings are those of `configs/official.yaml`: same architecture, data split, 102 training samples, optimiser and schedule. Within one seed all settings share the initial weights and the sample order. Only the residual and the weight vary.

| Stage | Config | Settings | Seeds | Runs |
|---|---|---|---|---|
| 1 (primary) | `full_mps` | data only, `shipped` 0.1, `consistent` 0.1 | 0, 1, 2 | 9 |
| 2 (weights) | `full_mps_sweep` | `shipped` and `consistent` at 0.001, 0.01, 1 | 0, 1, 2 | 18 |
| 3 (conditional) | `full_mps` with `name=full_mps_more "study.seeds=[3,4]"` | as stage 1 | 3, 4 | 6 |

Stages 1 and 2 are together exactly the 27 runs of `configs/full.yaml`, on a different device. Device: FNO on Apple MPS, residual on CPU. These are not CUDA results.

## Analysis

- **Primary comparison:** overall test relative L2 error, `consistent` 0.1 minus data-only, paired by seed, Student-t 95 % interval over seeds.
- **Secondary, descriptive:** the same for `shipped` 0.1; the other weights; error on constant-`Kcoeff` cells and near jumps; exact-operator and training residual ratios (log scale); train-test gap; change from 15 to 50 epochs for the same seed and setting; convergence at the last epochs.
- Secondary comparisons are additionally reported with a Holm correction over the settings compared on overall error. Only the primary comparison is confirmatory.
- Test-sample bootstrap intervals are not evidence about a setting (F14), and are not used for conclusions.

## Reading of the primary comparison

Stated in advance, in terms of the data-only mean error $E$:

- *Improvement* is claimed only if the upper end of the interval is below zero; *degradation* only if the lower end is above zero.
- *No material effect* is claimed only if the whole interval lies within $\pm 5\,\%$ of $E$. The margin is about twice the seed spread of the data-only error at 15 epochs (2.7 %).
- Anything else is *inconclusive*, and is reported as such.

## Stopping rule

Stage 3 runs if and only if the half-width of the primary interval after stage 1 exceeds 5 % of $E$. Results are then reported for $n = 3$ first and $n = 5$ second, since extending the sample after looking inflates error rates. No seeds are added after stage 3.

## Controls

- `scripts/check_determinism.py` before stage 1, to establish whether repeats on MPS are bit-identical and how far MPS and CPU diverge.
- Every run must finish with a finite loss, from a clean commit, with one shared hash of the training settings.
- Test error recomputed from every checkpoint.
