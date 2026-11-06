# Analysis plan for the operator, coefficient and label-budget studies (G3 to G6)

Written and committed before any run of `configs/full_mps_operator.yaml`, `full_mps_solver_w1.yaml`, `labels25*.yaml` or `labels50*.yaml` was trained. The commit that adds this file is the record of what was decided in advance; departures are reported as departures. It complements [`FULL_BUDGET_PLAN.md`](FULL_BUDGET_PLAN.md) (G2), whose stage-1 runs (`full_mps`: data only, `shipped` 0.1, `consistent` 0.1, seeds 0 to 2) are the controls here.

## Questions

1. **G3 (coefficient).** Does replacing the smoothed `Kcoeff` by the two-valued `coeff` in the cell-grid residual remove the higher error near permeability jumps that `consistent` 0.1 showed in every seed at 15 epochs (audit F9, F14)?
2. **G4 (operator).** Does a residual that the reference pressure satisfies exactly (the operator of the data generator with `coeff`, `pino_darcy.discrete`) improve the test error over data-only training, and does it avoid the jump-region degradation?
3. **G5, G6 (labels).** With 25 or 50 labelled samples, does the exact residual, on the labelled fields only or also on unlabelled training-split fields, improve the test error over data-only training with the same labels?

## Residuals compared

All in normalised units with the consistent forcing $1/(K_s U_s) = 57.21$, L1 reduction, factor $1/240 \times$ weight as upstream.

| Name | Operator | Coefficient | Reference residual (test set) |
|---|---|---|---|
| `consistent` | PhysicsInformer cell-grid finite differences, $h = 1/241$ | `Kcoeff` (model input) | 0.015 on constant cells, 166 near jumps (F4) |
| `consistent_coeff` | same | `coeff` | not zero near jumps (operator mismatch, F4 cause 2) |
| `solver_kcoeff` | generator scheme on the node grid, spline maps, $h = 1/240$ | `Kcoeff` | 26.5 mean (coefficient mismatch, F4 cause 1) |
| `solver` | same | `coeff` | about 0.07 mean in float32, 8 test samples (exact up to rounding) |

`consistent`, `consistent_coeff`, `solver_kcoeff`, `solver` form a 2 x 2 design: operator (cell grid, generator) x coefficient (`Kcoeff`, `coeff`). Nodes within 6 of the boundary are excluded from the `solver*` residuals (crop-and-extend, `reference_check.json`).

## Design

| Study | Config | Runs | Controls (same seed, same initial weights and sample order) |
|---|---|---|---|
| G3, G4 | `full_mps_operator` | `consistent_coeff`, `solver_kcoeff`, `solver` at 0.1; seeds 0, 1, 2 (9) | `full_mps`: data only, `consistent` 0.1 |
| G4, weight | `full_mps_solver_w1` | `solver` at 1; seeds 0, 1, 2 (3) | as above |
| G6 | `labels25`, `labels50` | data only and `solver` 0.1 with the first 25 or 50 training samples; seeds 0, 1, 2 (12) | each other |
| G5 | `labels25_unlabelled`, `labels50_unlabelled` | `solver` 0.1 plus the remaining 77 or 52 training-split fields as unlabelled PDE inputs; seeds 0, 1, 2 (6) | data only of `labels25`, `labels50` |

Every run has 5100 optimiser steps and the upstream per-step learning-rate schedule. Labelled samples are the first $N$ of the training split for every seed; the unlabelled fields are the remaining training-split samples and never come from validation or test. In the G5 runs every step adds one unlabelled field to the PDE term (an extra forward pass), so they cost more per step; the data term sees the same samples in the same order as in the corresponding `labels<N>` run. The data-only model at $N = 102$ and `solver` 0.1 at $N = 102$ are those of `full_mps` and `full_mps_operator`.

Code change between the controls (`eaca64c`) and these runs: the training residual is selected through `physics.PhysicsTerm`; for `shipped`, `consistent` and data-only, short CPU runs give bit-identical parameters at both commits (`results/audit/code_regression.json`).

The `solver` residual is chosen for G5 and G6 in advance, because it is the only one whose minimiser is the reference itself; it is not chosen from the G4 outcome.

## Analysis

Unit of replication: the training seed. Paired differences against the control of the same seed, Student-t 95 % interval over seeds (df = 2), as in `FULL_BUDGET_PLAN.md`.

- **Primary (G4):** overall test relative L2, `solver` 0.1 minus data-only.
- **Primary (G5, G6):** overall test relative L2, `solver` 0.1 minus data-only at $N = 25$ and at $N = 50$, with and without unlabelled fields: four comparisons, Holm-corrected.
- **Secondary, descriptive:** error on constant-`Kcoeff` cells and near jumps; exact residual (`evaluate_solver_residual.py`); G3 as `consistent_coeff` minus `consistent` near jumps; the 2 x 2 contrasts (coefficient effect at each operator, operator effect at each coefficient); `solver` 1 minus `solver` 0.1; train-test gap; computational cost per step.
- Reading of every interval as in `FULL_BUDGET_PLAN.md`: improvement only if the upper end is below zero, degradation only if the lower end is above zero, no material effect only if the interval lies within ±5 % of the control mean error, otherwise inconclusive.
- **Mechanistic prediction stated in advance.** If the jump-region degradation of `consistent` (F9, F14) is caused by the coefficient and operator mismatch of F4, the near-jump error difference to data-only should be positive for `consistent` and not positive for `solver`. If `solver` also degrades the near-jump error, the mismatch does not explain it.

No further seeds are added after looking at these results.
