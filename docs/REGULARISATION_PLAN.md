# Analysis plan: conventional regularisation against the corrected physics residual (G8)

Written and committed before any run of `configs/reg_cuda*.yaml`. The commit that adds this file is the record of what was decided in advance; departures will be reported as departures. One value is still open and is fixed by a rule stated here before any study run: the matched-compute budget B (section "Budgets"), measured in a timing pilot on the study device.

## Status of this study

This is a follow-up, not an independent confirmatory experiment. The physics-informed result it examines (`consistent` 0.1 lowers the test error of data-only training by 21.5 % [-23.5, -19.6] on CUDA and 21.6 % on MPS, five seeds; `SCIENTIFIC_AUDIT.md` F15, F18) was obtained on the same data, splits and test set. The physics weight 0.1 is the upstream default and was the pre-registered primary setting before any of those runs, but the earlier weight sweep also reported test errors, and they showed 0.1 to be the best weight tested. The design choices below (which controls, which grid, which caps) were made knowing those test results, and knowing the validation curves of the existing data-only runs (below). No test error of any run of this study is computed before the selection of the section "Selection" is written to `results/audit/regularisation_selection.json`.

## Question

Does training with the corrected residual (`consistent`, weight 0.1) give a lower test error than data-only training with conventional regularisation (weight decay, validation-based early stopping, or both), with the same data, splits, architecture, seeds and model-selection information, at equal optimiser steps and at matched compute?

The answer is not assumed. If a conventional control matches or beats the physics-informed model, that is the result.

## What the existing records show (validation and timing only)

- **Optimiser.** `engine.train_run` uses `torch.optim.Adam` (betas 0.9, 0.999, learning rate 1e-3, weight decay 0) with `ExponentialLR`, gamma 0.99948708 per optimiser step: the learning rate falls from 1e-3 to 7.31e-5 over 5100 steps. There is no weight decay, no early stopping, and only the final checkpoint (epoch 50) is kept. The validation error is recorded after every epoch (102 steps).
- **Validation curves.** In all ten data-only runs at the upstream budget (seeds 0 to 4, CUDA and MPS) the validation error is still falling at the end: the lowest validation error is at epoch 47 to 50. New minima late in training (epochs 30 to 50) come every 2 epochs on average, with gaps of up to 7 epochs. Applied to the CUDA data-only histories (which the λ = 0 runs of this study should nearly reproduce), the rules below give: early stopping with patience 5 stops only seed 0 (at epoch 46, selecting epoch 41, validation error 0.04304 against 0.04285 at epoch 50); otherwise both rules select epoch 47 to 50. Validation-based selection is therefore not expected to act as a strong regulariser at this budget. This is stated before the study so that a null result is not over-read.
- **Cost of the residual.** On the Tesla T4 (F18) a data-only run takes 189.5 to 190.6 s and a `consistent` 0.1 run 192.9 to 193.9 s end to end; of these, 71 s is the per-epoch validation, which both do. Optimisation alone: 23.2 against 23.9 ms per step. The residual adds about 2 % end to end and 3 % to the optimisation, so the matched-compute budget of the data-only controls is expected to be about one epoch more than 50. The pilot below measures it with the new code instead of assuming it.

## Controls and models

All data-only models use the configuration of `full_cuda` (data, splits, FNO, batch size 1, start learning rate, gamma, validation after every epoch) with three additions; the physics-informed model is not retrained.

| Arm | Training | Model used |
|---|---|---|
| base | data only, AdamW with weight decay 0 (identical to the upstream Adam, tested bitwise on CPU) | model at the budget |
| WD | data only, AdamW with the selected weight decay `wd*` | model at the budget |
| ES | data only, weight decay 0, early stopping (rule `es`) | model selected by early stopping within the budget |
| VB | data only, weight decay 0, best validation epoch (rule `vb`) | model of the lowest validation error within the budget |
| WD+ES | data only, `wd*`, rule `es` | as ES |
| WD+VB | data only, `wd*`, rule `vb` | as VB |
| PINO | `consistent` 0.1, upstream Adam, 5100 steps: `consistent_w0.1_seed<s>` of `full_cuda`, `full_cuda_more` (F18), existing checkpoints | final model |

The ES and VB arms do not need runs of their own: early stopping depends only on the validation errors up to the current epoch, so a run trained past its stopping epoch passes through exactly the state a halted run would have. `engine.train_run` keeps, for each cap, the weights early stopping selects and records the epoch, the stop and the compute consumed until the stop (`tests/test_regularisation.py` checks the rule, the tracker against it, and that every snapshot equals the final model of a run stopped there).

**Weight decay: AdamW, decoupled.** `torch.optim.Adam(weight_decay=λ)` adds `λθ` to the gradient, which Adam then divides by the running root-mean-square gradient: parameters with large gradient variance are hardly regularised, and the effective strength changes during training. `torch.optim.AdamW` multiplies the weights by `1 - lr·λ` before the Adam step (Loshchilov and Hutter, 2019): a shrinkage that follows the learning-rate schedule and is the same for every parameter. It is the form of weight decay used with Adam in current practice and has a direct reading here: with the upstream schedule the sum of the learning rates over 5100 steps is 1.806, so decay alone would shrink the weights by a factor `exp(-1.806 λ)` over training. The coupled L2 penalty is not studied; its result could differ.

**Grid.** λ in {0, 1e-3, 1e-2, 1e-1, 1, 10}, one decade apart. Decay alone would shrink the weights over the whole run by 0.2 %, 1.8 %, 16 %, 84 % and a factor 7e-9 at the five non-zero values, so the grid runs from negligible to dominant. Boundary rule: if the selected λ is 10, the grid is extended once to 100 (`configs/reg_cuda_wd1e2.yaml`) and the selection repeated; there is no further extension. If 0 has a lower validation error than every non-zero value, `wd*` is still the best non-zero value (the WD arms then show the effect of the least harmful decay) and this is reported.

**Validation-based selection**, two rules, fixed in advance and not tuned. Both use the validation relative L2 on the 102 validation samples after every epoch (102 optimiser steps); an epoch improves if its validation error is below the lowest so far (minimum improvement 0); the selected model is that of the lowest validation error up to the stop or the cap; caps are the two budgets.

- `es`, early stopping: training stops at the first epoch at which 5 epochs have passed since the last improvement. Patience 5 is 10 % of the budget and 2.5 times the mean spacing of new minima late in the existing runs. It saves compute when it stops, and can stop on a temporary plateau.
- `vb`, best validation checkpoint: no stop; the epoch of the lowest validation error within the cap. It uses the whole budget and is the strongest use of the validation split available to a data-only model under the budget.

## Budgets

**A, equal optimiser steps:** 5100 steps (epoch 50) for every arm. ES arms: at most 5100 steps; their actual steps and time are reported.

**B, matched compute:** the data-only arms may use the end-to-end training time of the physics-informed run. Rule, fixed now: in a pilot on the study device with the code of the study, train seed 0 for 3 epochs with (i) `reg_cuda_wd1e-2` (AdamW, early-stopping bookkeeping) and (ii) `consistent` 0.1 as in `full_cuda`; take the mean time of epochs 2 and 3 (`elapsed_seconds` in `history.csv`, including validation) as `t_data` and `t_pino`; then

  `B = ceil(50 · t_pino / t_data)` epochs,

written into `configs/reg_cuda.yaml` (`max_epochs`) and committed before any study run. The learning-rate schedule is not stretched: the extra epochs continue the upstream per-step decay, so the first 50 epochs of every run are those of budget A and one run per (λ, seed) gives both budgets. ES arms at B: at most B epochs; actual consumption reported. The provisional value 51 in the configuration follows from the F18 timings (`ceil(50 · 193.5 / 190.0) = 51`) and is replaced by the pilot's value.

Equal epochs are not claimed to be equal compute; the times actually consumed are reported for every arm.

## Selection (validation only)

1. Train all grid studies (`reg_cuda_wd0` to `reg_cuda_wd1e1`, seeds 0 to 4).
2. `scripts/evaluate_variants.py --stage validation` for every grid study: training and validation errors of every saved model. The test split is not read.
3. `scripts/regularisation_study.py select`: `wd*` is the non-zero λ with the lowest mean validation relative L2 over seeds 0 to 4 of the budget-A model; the same `wd*` is used for budget B and for WD+ES. The file records `"test_set_read": false` and is committed (or archived) before step 4. If the boundary rule applies, the file has status `extend`; the extension is trained, evaluated on validation, and `select` run once more.
4. Only then `scripts/evaluate_variants.py --stage test` (it refuses to run without the selection file and stores its hash), for the grid studies and for `full_cuda`, `full_cuda_more` (the physics-informed runs, final model only).

The same seeds are used for selection (validation split) and for testing (test split); the test split is never used for any choice. Selecting from six values by validation error can make the selected arm look better on validation than it is; its test error is not affected by that choice.

## Primary comparison and statistics

**Primary:** budget A, PINO minus WD+VB, test relative L2 (mean over the 256 test samples per model), paired by seed over seeds 0 to 4; mean difference with a 95 % Student-t interval (4 degrees of freedom), relative to the mean error of WD+VB. WD+VB is fixed in advance as the strongest conventional control: weight decay chosen on validation and the best validation checkpoint within the same step budget. It is fixed rather than chosen among the controls by validation error, because the validation-selected checkpoints are selected on that same split and would be favoured by such a choice. It is preferred to WD+ES because patience-based stopping can only select a model that `vb` could also select, at a validation error no lower; the physics-informed model gets no validation-based selection at all, so the primary comparison gives the control every advantage the validation split offers.

**Reading** of an interval of a difference (setting minus comparison): *setting better* if it lies below zero, *comparison better* if above, *no material difference* if it lies entirely within ±5 % of the comparison mean, otherwise *inconclusive*. A non-significant difference is not read as equivalence; only the ±5 % rule says "no material difference".

**Secondary**, Holm-adjusted together (21 comparisons, test relative L2): at budget A, each of WD, ES, VB, WD+ES, WD+VB minus base, and PINO minus base, WD, ES, VB and WD+ES; at budget B, the same five controls minus base, and PINO minus base and each of the five controls.

**Descriptive**, per arm and budget: mean and per-seed test error; training and validation error; error near permeability jumps and on constant-`Kcoeff` cells; residual under the operator the reference solves (as F16); training time and optimiser steps actually used; selected epoch. Paired differences PINO minus base and PINO minus WD+ES for the near-jump error and the solver residual. Early-stopping epochs and the compute they save. A check of the re-trained base against the data-only runs of F18 (same seeds, same device, AdamW with λ = 0 is the upstream update): the per-seed difference is reported.

**Interpretation limits** stated in advance: the comparison is at the upstream learning rate and schedule for every arm (no learning-rate tuning for any arm); the PINO weight is not tuned on validation while λ is, which favours the controls; five seeds; one dataset, one architecture, one GPU model. A smaller generalisation gap or residual is a correlation with the training method, not a demonstration of a mechanism.

## Runs, compute and storage (estimates)

| Item | Runs | Estimate on a Tesla T4 |
|---|---|---|
| Timing pilot (separate session) | 2 runs of 3 epochs | about 15 min including set-up |
| Grid, 6 values × 5 seeds, B ≈ 51 epochs | 30 | 30 × about 195 s ≈ 1.6 h |
| Variant evaluation (4 models per run, validation and test), PINO evaluation | 30 + 10 | about 0.3 h |
| Grid extension, if the boundary rule applies | 5 | about 0.3 h |
| Total main session | 30 (35) | about 2 to 2.5 h, within one Kaggle session |

Estimates from the F18 timings; the pilot replaces them. Storage, outside Git: per run the final checkpoint (28 MB) and five snapshots (`epoch050` and the two rules at the two caps, 9 MB each; identical selections are still stored separately), about 75 MB; 2.3 GB for 30 runs (2.6 GB with the extension), about the same again for archives in the session. Tracked records: about 2 MB per grid study plus the variant files (per-sample test errors), about 15 MB in all.

## Execution

On a machine with one NVIDIA GPU, from the repository root at the commit that fixes B (on Kaggle, through the launcher once it has a plan for this study, with the `full_cuda` and `full_cuda_more` eval bundles restored for the physics-informed checkpoints):

```bash
for s in reg_cuda_wd0 reg_cuda_wd1e-3 reg_cuda_wd1e-2 reg_cuda_wd1e-1 reg_cuda_wd1e0 reg_cuda_wd1e1; do
  python scripts/train.py --config $s --resume                                 # trains, then evaluates the final model
  python scripts/evaluate_variants.py --config $s --stage validation
done
python scripts/regularisation_study.py select                                  # validation only; freezes wd*
for s in reg_cuda_wd0 reg_cuda_wd1e-3 reg_cuda_wd1e-2 reg_cuda_wd1e-1 reg_cuda_wd1e0 reg_cuda_wd1e1 full_cuda full_cuda_more; do
  python scripts/evaluate_variants.py --config $s --stage test
done
python scripts/regularisation_study.py analyse                                 # results/audit/regularisation_study.json
```

`full_cuda` and `full_cuda_more` also need `--stage validation` (training and validation errors of the physics-informed runs) before `analyse`; that step does not read the test split and may run at any time.

Pilot (fixes B):

```bash
python scripts/train.py --config reg_cuda_wd1e-2 --no-evaluate name=pilot_reg_wd max_epochs=3 "snapshots.epochs=[2]" "early_stopping.caps=[2,3]" "study.seeds=[0]"
python scripts/train.py --config full_cuda --no-evaluate name=pilot_reg_pino max_epochs=3 "study.seeds=[0]" "study.residuals=[consistent]" "study.physics_weights=[0.1]"
```
