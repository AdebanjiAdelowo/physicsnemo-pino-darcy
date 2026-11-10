# Analysis plan: conventional regularisation against the corrected physics residual (G8)

Written and committed before any run of `configs/reg_cuda*.yaml`. The commit that adds this file is the record of what was decided in advance; departures will be reported as departures. One value was left open and fixed by a rule stated here before any study run: the matched-compute budget B (section "Budgets"), measured in a timing pilot on the study device. The pilot has run (B = 52 epochs), and one amendment was made after it and before any study run (section "Amendment 1").

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

written into `configs/reg_cuda.yaml` (`max_epochs`) and committed before any study run. The learning-rate schedule is not stretched: the extra epochs continue the upstream per-step decay, so the first 50 epochs of every run are those of budget A and one run per (λ, seed) gives both budgets. ES arms at B: at most B epochs; actual consumption reported. The provisional value 51 in the first version of the configuration followed from the F18 timings (`ceil(50 · 193.5 / 190.0) = 51`); the pilot gave 52, which is the value used (Amendment 1).

Equal epochs are not claimed to be equal compute; the times actually consumed are reported for every arm.

## Amendment 1: budget B from the pilot, and an epoch-51 sensitivity analysis (2026-11-09)

Made after the timing pilot and before any study run and any test evaluation of this study; approved before training. Nothing else in this plan changes: the primary comparison (budget A), the weight-decay grid and its boundary rule, the selection rules, the intervals and the reading rule are as above.

**Pilot result** (Kaggle session `pino-darcy-reg-pilot-cd7ca41`, Tesla T4, commit `cd7ca41`, `results/audit/regularisation_pilot.json`, committed unchanged): epochs 2 and 3 took 3.715 and 3.679 s (`reg_cuda_wd1e-2`) and 3.801 and 3.797 s (`consistent` 0.1), so `t_data` = 3.697 s, `t_pino` = 3.799 s, ratio 1.0277 and **B = ceil(51.39) = 52 epochs**, by the pre-registered rule. The first three epochs of AdamW without decay and of `consistent` 0.1 were identical to those of `full_cuda` (relative difference 0), so the runs are comparable with the historical ones. The environment matched `full_cuda` exactly.

**Why an amendment.** The ratio lies 0.0077 above 1.02, where B changes from 52 to 51, and the data-only epoch times vary by about ±1.6 % between epochs. The same two-epoch estimator applied to the five full-length runs of F18 gives B = 53, 51, 52, 52 and 51, and the mean over their epochs 2 to 50 gives a ratio of 1.019 (B = 51). The matched-compute budget is therefore at the edge between 51 and 52 epochs, and the rule decides between them by chance. The rule is kept, and its value 52 is the official budget B. Changing the estimator after seeing its result is not done.

**What is added.** Every run also keeps the model after epoch 51 (`epoch051`, 5202 optimiser steps) and the models both rules select with a cap of 51 epochs (`es_cap051`, `vb_cap051`), from the same 52-epoch runs: no extra training. The analysis reports a sensitivity budget **B51** with the same arms and comparisons as budget B, unadjusted and outside the Holm family, next to the official budget B (52 epochs). The choice between 51 and 52 is never made on test errors: 52 is official, B51 is reported whatever it shows. The selection file lists, from validation errors only, the epoch each rule selected with each cap in every run.

**Boundary extension.** If the boundary rule asks for the weight decay 100, the `train_select` session stops with the completed runs archived (`AUTO_EXTEND = False` in the launcher) and the extension is trained only after a decision.

## Selection (validation only)

1. Train all grid studies (`reg_cuda_wd0` to `reg_cuda_wd1e1`, seeds 0 to 4).
2. `scripts/evaluate_variants.py --stage validation` for every grid study computes training and validation errors of every saved model. The completed run loaded the test split as described below. The remediated validation path loads only training and validation tensors.
3. `scripts/regularisation_study.py select` chooses `wd*`, the non-zero λ with the lowest mean validation relative L2 over seeds 0 to 4 of the budget-A model. The same `wd*` is used for budget B and for WD+ES. If the boundary rule applies, the file has status `extend` and selection is repeated once after validation of the extension.
4. After authorization, `scripts/evaluate_variants.py --stage test` requires the committed freeze and a verified restore, and records the selection digest. It covers the grid studies and the final models of `full_cuda` and `full_cuda_more`.

The same five seeds are used for selection and testing. The registered selection inputs are validation errors. Choosing from six values can make the selected arm look better on validation through selection on that split.

### Holdout-access deviation recorded after training (2026-11-09)

The completed Kaggle workflow at `1286cbcaafbe21eda2dac8c0549c4a0e69243a3e` opened test tensors before selection through the shared dataset loader. Preflight also computed descriptive statistics and reference-solution residuals on the test split. The inspected training and checkpoint-selection paths use training and validation data, and the archived validation histories independently reproduce the frozen decisions. The reviewed logs and artifacts contain no new test-set model evaluation. They cannot establish who saw the diagnostics or exclude activity outside the recorded workflow.

The original export's `test_set_read: false` field is inaccurate as a record of literal access. The byte-identical export is preserved with SHA-256 `1e9bbc8ffe07822035e7aa551da8653c961f7b1fa488741138013cf698963c7c`. The [scientific audit](SCIENTIFIC_AUDIT.md#f19-g8-holdout-access-and-selection-freeze) records the deviation separately. This is a follow-up on a previously examined test set. The selected decay of 0.001, the paired seeds, the 50- and 52-epoch budgets, B51 as a sensitivity analysis, and the primary Budget A PINO versus WD+VB comparison remain as registered.

## Primary comparison and statistics

**Primary:** budget A, PINO minus WD+VB, test relative L2 (mean over the 256 test samples per model), paired by seed over seeds 0 to 4; mean difference with a 95 % Student-t interval (4 degrees of freedom), relative to the mean error of WD+VB. WD+VB is fixed in advance as the strongest conventional control: weight decay chosen on validation and the best validation checkpoint within the same step budget. It is fixed rather than chosen among the controls by validation error, because the validation-selected checkpoints are selected on that same split and would be favoured by such a choice. It is preferred to WD+ES because patience-based stopping can only select a model that `vb` could also select, at a validation error no lower; the physics-informed model gets no validation-based selection at all, so the primary comparison gives the control every advantage the validation split offers.

**Reading** of the 95 % interval of a difference (setting minus comparison), in this order: *setting better* if the whole interval lies below zero; *comparison better* if it lies above zero; *no material difference* if it lies entirely within ±5 % of the comparison mean; otherwise *inconclusive*.

**The ±5 % band.** It is a practical-relevance margin, kept from [`FULL_BUDGET_PLAN.md`](FULL_BUDGET_PLAN.md) so that this study reads its intervals as the earlier ones did; it is not derived from these data. "No material difference" is an interval-inclusion statement: the whole 95 % interval inside the band, which is the same as two one-sided tests at 2.5 % each, stricter than the common 90 % interval. It says that a difference larger than 5 % of the comparison mean is excluded at that confidence; it does not say that the methods are identical. A difference whose interval includes zero but extends beyond the band is *inconclusive*: failure to detect a difference is never reported as equivalence, and a p value above 0.05 is never read as "no difference". For scale (F18, CUDA, already reported): 5 % of the data-only mean error is 0.0019; the seed-to-seed SD of the data-only error is 0.0013 (3.3 %); the paired PINO minus data-only differences had an SD of 0.0006, which with five seeds gives a 95 % half-width of 0.0007 (1.9 % of the mean). The band can therefore be met with five seeds if two arms differ by little and their per-seed differences are as consistent as in F18; if they are less consistent, "inconclusive" is the expected reading.

**Secondary**, Holm-adjusted together (21 comparisons, test relative L2): at budget A, each of WD, ES, VB, WD+ES, WD+VB minus base, and PINO minus base, WD, ES, VB and WD+ES; at budget B, the same five controls minus base, and PINO minus base and each of the five controls.

**Descriptive**, per arm and budget: mean and per-seed test error; training and validation error; error near permeability jumps and on constant-`Kcoeff` cells; residual under the operator the reference solves (as F16); training time and optimiser steps actually used; selected epoch. Paired differences PINO minus base and PINO minus WD+ES for the near-jump error and the solver residual. Early-stopping epochs and the compute they save. A check of the re-trained base against the data-only runs of F18 (same seeds, same device, AdamW with λ = 0 is the upstream update): the per-seed difference is reported.

**Interpretation limits** stated in advance: the comparison is at the upstream learning rate and schedule for every arm (no learning-rate tuning for any arm); the PINO weight is not tuned on validation while λ is, which favours the controls; five seeds; one dataset, one architecture, one GPU model. A smaller generalisation gap or residual is a correlation with the training method, not a demonstration of a mechanism.

## Selection bias and leakage audit

**Paths from the test split to a choice, and how each is closed.**

| Path | Closed by |
|---|---|
| Training evaluates the final model on the test split (`train.py` without `--no-evaluate`) | The launcher trains with `--no-evaluate`. The gated evaluation entry points check the committed receipt and frozen evidence, the effective study configuration, the restored checkpoint inventory and its hashes, and the dataset identifiers and file hashes before loading test tensors. |
| Weight decay chosen by test error | `select` reads only the `*_variants_validation.json` files. The remediated training and validation loaders omit test tensors. Dataset preparation can still write and hash all splits for identity verification. |
| Checkpoint chosen by test error | The `es` and `vb` rules see only the per-epoch validation error during training. |
| Selection repeated after test results are seen | `select` refuses to overwrite a final selection. The gate authenticates the original selection, reports and manifests against the pinned receipt and their exact bytes at HEAD. An alternate or edited JSON cannot pass by claiming final status. |
| Changed plan after test results | This file, the grid, the rules, the primary comparison and the analysis code are committed before any study run; budget B is set by a stated formula from the pilot, which reads no test data. |

**Remaining sources of bias, stated in advance.**

1. *Same seeds for selection and testing.* The weight decay and the validation-best checkpoints are chosen on the validation split using the five trained models whose test errors are then reported. Validation and test errors of one model are correlated (same model, same input distribution), so the selected arms' test errors are somewhat optimistic compared with a new set of seeds trained with the selected setting. This favours the conventional controls. A PINO advantage that survives is therefore conservative; a finding that a control matches PINO must be read with this optimism in mind. To show its size, the analysis reports the validation and test errors of every grid value side by side (after the selection).
2. *Asymmetric tuning.* The controls get a tuned weight decay and validation-based checkpoint selection; the physics-informed model gets neither (weight 0.1 fixed in advance, final checkpoint). This also favours the controls.
3. *Knowledge of earlier results.* The design was made knowing the published test results of F15 and F18 and the validation curves of the existing runs (section "Status of this study"). The grid follows from the learning-rate schedule, not from any error; the patience was set knowing the validation curves.
4. *Arms that share runs.* ES, VB and the fixed-step model of one weight decay come from the same run, and base and the physics-informed run share the seed. All comparisons are paired by seed, with seeds as the independent units. Where a rule selects the final model in every seed, a difference is exactly zero with zero spread; its interval is then the point zero, and it is reported as such rather than as evidence about variability.
5. *Multiplicity.* One primary comparison; the 21 secondary comparisons are Holm-adjusted together; the regional, residual and compute comparisons are descriptive and unadjusted.

## Comparability with the historical baseline

The base arm (AdamW, weight decay 0) is trained anew rather than reusing the data-only runs of F18, so that every data-only arm comes from the same code path and session. It is comparable with the physics-informed runs it is tested against because:

1. *Same update.* `torch.optim.AdamW` with weight decay 0 performs the same arithmetic as the upstream `torch.optim.Adam`; `tests/test_regularisation.py` checks bitwise equality on CPU.
2. *Same software.* `kaggle/regularisation_constraints.txt` pins PyTorch 2.15.0, NumPy 2.5.4, Warp 1.19.0 and SymPy 1.14.0, the versions recorded for `full_cuda`, and `regularisation_session.py env-check` stops the session unless the installed PyTorch (including its CUDA build, 13.2), NumPy, Warp, SymPy, Python minor version and PhysicsNeMo commit equal those of `results/full_cuda/study_metadata.json`. The complete package list and the GPU and driver are saved with every session.
3. *Same trajectory, checked.* The pilot trains seed 0 for three epochs with AdamW without decay and with the physics-informed configuration, and compares the per-epoch data loss and validation error with those of `full_cuda` (validation only). They must agree to 1e-3 relative (the level at which CUDA and MPS agreed, F18); otherwise `check-pilot` stops the study before any training.
4. *After the study*, the 50-epoch base runs are compared per seed with the F18 data-only runs (test error and history), and the difference is reported.

Not controlled: the physical GPU and driver of a Kaggle session (recorded), and non-deterministic CUDA kernels; both were present in F18 as well.

## Runs, compute and storage (estimates)

| Session | Work | Estimate on a Tesla T4 |
|---|---|---|
| `pilot` | set-up and checks (about 10 min), restore of the inputs, 3 runs of 3 epochs | about 15 min |
| `train_select` | set-up, 30 runs of B ≈ 51 epochs (30 × about 195 s ≈ 1.6 h), validation evaluation of 6 models per run and of the 10 physics-informed runs (about 0.2 h); with the grid extension 5 more runs (about 0.3 h) | about 2 to 2.3 h |
| `test` | set-up, restore, test evaluation of every model and the summaries, analysis | about 0.4 h |

Estimates from the F18 timings; the pilot replaces them. Storage, outside Git: per run the final checkpoint (28 MB) and eight snapshots (`epoch050`, `epoch051` and the two rules at the three caps, 9 MB each; identical selections are still stored separately), about 100 MB; 3.0 GB for 30 runs (3.5 GB with the extension), about the same again for archives in the session. Tracked records: about 2 MB per grid study plus the variant files (per-sample test errors), about 15 MB in all.

## Execution

Three Kaggle sessions with `kaggle/run_regularisation.ipynb` (the replication launcher `run_cuda.ipynb` is not used). Each checks out one commit by its full SHA, installs the pinned environment, checks it against `full_cuda`, checks the dataset and runs the test suite. Attach the output of `pino-darcy-cuda-minimal-416bfe6` as an input in every session.

1. `MODE = "pilot"`: restores and verifies the physics-informed inputs (`kaggle/regularisation_inputs.sha256`: 4 archives, 30 checkpoint files, metadata) and runs the pilot (3 runs of 3 epochs). Output `regularisation_pilot.json`. Then, locally: commit it as `results/audit/regularisation_pilot.json` and set `max_epochs` in `configs/reg_cuda.yaml` to its `budget_B_epochs` (a test checks that they agree).
2. `MODE = "train_select"` at that commit: trains the six grid studies with `--resume --no-evaluate`, archives each when finished (records without test results, checkpoints and snapshots, SHA-256 manifest), evaluates every model on the training and validation splits, and runs `select` (and the extension, if the boundary rule asks for it). A session that runs short stops before a study it cannot finish; the next one continues from this output. Then, locally: commit `regularisation_selection.json`, the `*_variants_validation.json` files and `regularisation_archives.sha256` to `results/audit/`.
3. After separate authorization, `MODE = "test"` at the remediation commit, with the `train_select` output attached. Verify the committed freeze, restore against the committed manifests in a clean checkout, verify the restored inputs, then run evaluation and analysis. The gate verifies the files used by each evaluation again.

The same steps without the launcher, on any machine with one NVIDIA GPU and the pinned environment:

```bash
python scripts/regularisation_session.py pilot --out regularisation_pilot.json
# commit the pilot, set budget B, then:
for s in reg_cuda_wd0 reg_cuda_wd1e-3 reg_cuda_wd1e-2 reg_cuda_wd1e-1 reg_cuda_wd1e0 reg_cuda_wd1e1; do
  python scripts/train.py --config $s --resume --no-evaluate
  python scripts/evaluate_variants.py --config $s --stage validation
done
python scripts/evaluate_variants.py --config full_cuda --stage validation
python scripts/evaluate_variants.py --config full_cuda_more --stage validation
python scripts/regularisation_study.py select                                  # validation only; freezes wd*
# for the completed study, use the committed remediation checkout and prepare the dataset
python scripts/prepare_data.py
python scripts/regularisation_session.py check-selection
python scripts/regularisation_session.py restore-inputs --input-root /path/to/preserved/outputs
python scripts/regularisation_session.py restore-runs --input-root /path/to/preserved/outputs --manifest results/audit/regularisation_archives.sha256
python scripts/regularisation_session.py check-restore
# after separate authorization:
for s in reg_cuda_wd0 reg_cuda_wd1e-3 reg_cuda_wd1e-2 reg_cuda_wd1e-1 reg_cuda_wd1e0 reg_cuda_wd1e1 full_cuda full_cuda_more; do
  python scripts/evaluate_variants.py --config $s --stage test
done
python scripts/regularisation_study.py analyse                                 # results/audit/regularisation_study.json
```
