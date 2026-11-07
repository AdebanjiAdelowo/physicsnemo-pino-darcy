# Scientific and software audit

Audit date 2026-10-24. Object: this repository at commit `6ce3229` (the state before the audit), its tracked results (`results/data_check.json`, `results/local/`), and its relation to the NVIDIA PhysicsNeMo example `examples/cfd/darcy_physics_informed` at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`. The upstream comparison is in [`UPSTREAM_COMPARISON.md`](UPSTREAM_COMPARISON.md), missing experiments in [`EXPERIMENT_GAPS.md`](EXPERIMENT_GAPS.md), the checklist in [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md).

Classification:

- **VERIFIED**: reproduced in this audit from code, data or result files, or proven.
- **SUPPORTED BUT LIMITED**: consistent with the evidence, but the evidence cannot carry the general claim (one seed, reduced budget, test-sample variability only).
- **UNVERIFIED**: not checked, or not checkable with what exists.
- **INCORRECT**: contradicted by the evidence.

Everything measured in the original audit is CPU or Apple MPS inference and NumPy/SciPy computation; no model was trained for it. The addendum F14 (same date) adds six training runs, G1 of `EXPERIMENT_GAPS.md`, and updates F6 to F9 where they bear on seeds; the seed-0 findings are kept as they were.

## Summary

The implementation reproduces the upstream example faithfully, the tracked numbers are what the code produced, and the README states its limits accurately. The audit confirms the two residual corrections and adds one finding the project had not identified: the reference pressure was generated from the two-valued coefficient `coeff` with a specific discrete scheme on a staggered node grid, while both residual definitions use the smoothed model input `Kcoeff` on the cell-centred grid. This explains the large residual of the reference near permeability jumps, it qualifies the name "data-consistent", and it gives an exact residual against which the trained models can be measured. Under that exact residual the physics-weighted models still have a lower residual than the data-only model, so the main residual finding stands; the accuracy finding remains a single-seed observation with no established improvement.

| ID | Claim or question | Verdict |
|---|---|---|
| F1 | Upstream forcing 0.01748 is wrong; the scaled equation needs 57.21 | VERIFIED |
| F2 | Stored fields are cell-centred with spacing 1/241, not 1/240 | VERIFIED |
| F3 | What the reference pressure solves | VERIFIED (new) |
| F4 | The `consistent` residual is consistent with the data | SUPPORTED BUT LIMITED: only away from jumps |
| F5 | The cell-grid residual is a valid diagnostic near jumps | INCORRECT as a measure of PDE violation there; the README already warns about this |
| F6 | `consistent`, weight 0.1, is as accurate as data-only (0.0475 against 0.0480) | VERIFIED as "no detectable difference", now over three seeds (F14); any improvement UNVERIFIED |
| F7 | `shipped` raises the error from weight 0.01 upwards | SUPPORTED BUT LIMITED: at 0.1 worse in 3 of 3 seeds, interval over seeds just includes zero (F14); other weights one seed |
| F8 | Physics weighting lowers the PDE residual | VERIFIED at weight 0.1 for three seeds, under the training and the exact residual (F14); other weights one seed |
| F9 | Lower residual is not closer agreement with the reference | VERIFIED; the higher error near jumps at weight 0.1 reproduces in 3 of 3 seeds (F14) |
| F10 | Splits are disjoint, no leakage | VERIFIED |
| F11 | Tracked summaries match the per-run records | VERIFIED, with one labelling issue (median) |
| F12 | Implementation matches upstream | VERIFIED |
| F13 | Smaller README statements | one INCORRECT figure (corrected), others VERIFIED |
| F14 | Seed study at weight 0.1 (G1, addendum) | residual reduction REPRODUCED; no accuracy improvement; see section |

## F1. Forcing term

**Claim (README).** In the normalised variables the data satisfy $-\nabla\cdot(k'\nabla u') = 57.2$; the upstream script uses 0.0175.

**Derivation.** With $k = K_s k'$ and $u = U_s u'$, $-\nabla\cdot(k\nabla u) = 1$ becomes $K_s U_s\,(-\nabla\cdot(k'\nabla u')) = 1$, so the normalised forcing is $1/(K_s U_s) = 1/(4.49996 \times 3.88433\times10^{-3}) = 57.2104$. The upstream line `forcing_fn = 1.0 * 4.49996e00 * 3.88433e-03` is the product, 0.0174793.

**Evidence.**

- Upstream source read at the pinned commit (`darcy_physics_informed_fno.py`, line `forcing_fn = ...`).
- The physical forcing of the data is 1: the original solver is called with `f = ones` (`demo.m`), and on cells with locally constant permeability $-k\,\Delta_h u = 1.0000015$ (standard deviation $3.3\times10^{-4}$, 460 samples, `results/data_check.json`).
- `tests/test_reference.py::test_normalised_forcing_is_the_reciprocal_of_the_scale_product` solves the scaled problem directly: the solution with coefficient $k/K_s$ and forcing $1/(K_s U_s)$ equals $u/U_s$ to $10^{-10}$; with the shipped forcing the solution is $3.06\times10^{-4}$ times smaller.
- Consequence: under the shipped residual the reference has $|r| = 56.72$ on every cell of constant permeability (`data_check.json`, `abs_mean_smooth`), the residual of a field that solves the PDE exactly.

**Verdict: VERIFIED.** The correction is mathematically justified and empirically confirmed. The upstream value is incorrect for this dataset.

## F2. Grid spacing

**Claim.** The stored fields are on cell centres with spacing 1/241; upstream uses 1/240.

**Evidence.**

- The original solver `solve_gwf.m` places the coefficient on `X1 = 1/(2K) : 1/K : (2K-1)/(2K)`, the cell centres $(i + 1/2)/241$, and returns the solution there. Its internal node grid `X2 = 0 : 1/(K-1) : 1` has spacing 1/240. F3 shows that the dataset was produced by this procedure.
- Independent checks already in `data_check.json`: row-2 to row-1 pressure ratio next to the wall 2.971 (cell-centred grid: about 3); $-k\Delta_h u = 1.0000$ with $h = 1/241$ and $0.9917$ with $h = 1/240$.

**Verdict: VERIFIED.** The upstream comment "unit square with resolution as 240" describes the solver's internal grid, not the stored fields. The effect on second differences is 0.8 %, small next to F1. The 240 x 240 crop keeps cell centres 0 to 239 and drops the last row and column, so the cropped grid is not symmetric about the domain centre; this matters only for boundary-aware residuals (F3, cropped evaluation).

## F3. What the reference pressure satisfies (new)

**Question.** The README establishes that no five-point stencil on the stored grid reproduces the reference near permeability changes. What does the reference satisfy?

**Method.** `src/pino_darcy/reference.py` re-implements the FNO data-generation script `data_generation/darcy/solve_gwf.m` (Li et al., 2021). Source: the copy in the repository `li-Pingan/fourier-neural-operator` (commit `f01cf052de7899c642ba738b8a322bdd488d9d38`), retrieved through the GitHub API because the original repository has moved; it is a third-party copy, so its identity with the script used for this dataset is established only by the regeneration below. The procedure:

1. Interpolate the cell-centred coefficient to the 241 x 241 nodes $i/240$ with a not-a-knot cubic spline along each axis (MATLAB `interp2(..., 'spline')`).
2. Solve $-\nabla_h\cdot(a\nabla_h p) = 1$ on the interior nodes with the conservative five-point scheme, arithmetic face means, $h = 1/240$, $p = 0$ on the boundary nodes.
3. Interpolate $p$ back to the cell centres with the same spline.

`scripts/verify_reference.py` applies it to all 460 samples and writes `results/audit/reference_check.json`.

**Results.**

| Input coefficient | Relative L2 difference from the stored `sol` (mean, max over 460 samples) |
|---|---|
| `coeff` (two-valued, 3 and 12) | $2.47\times10^{-8}$, $2.74\times10^{-8}$ |
| `Kcoeff` (smoothed, the model input) | $1.71\times10^{-2}$, $5.09\times10^{-2}$ |

$2.5\times10^{-8}$ is the rounding level of the float32 storage, so the stored pressure is exactly `solve_gwf(coeff, 1)`.

Residual of the stored pressure under the discrete operator of the solver (`reference.solver_residual`: map the cell values to the nodes by inverting the spline, apply the scheme), physical units, forcing 1:

| Coefficient | Mean abs | Max abs | Interface nodes | Other nodes |
|---|---|---|---|---|
| `coeff` | 2.85e-04 | 4.54e-03 | 3.33e-04 | 2.84e-04 |
| `Kcoeff` | 4.77e-01 | 4.19e+02 | 1.02e+01 | 1.66e-01 |

Interface nodes: 3.1% of interior nodes (within two cells of a jump of `coeff`). After cropping to 240 x 240 and extending back (as needed for model outputs), the mean and maximum residual beyond a margin of 4 nodes are 2.92e-04 and 7.08e-03; at a margin of 2 the maximum is 2.20e+00.

For comparison, the best cell-grid stencil of `data_check.json` leaves a mean absolute residual of 2.84 on cells with varying permeability.

**Verdict: VERIFIED.** The reference is an exact solution of a known discrete problem everywhere, including at jumps. Its large finite-difference residual near jumps (F5) is a mismatch of operators (grid, coefficient, interpolation), not an inaccuracy of the data. `tests/test_reference.py` checks the solver, its residual and the spline on synthetic fields.

## F4. "Data-consistent" residual

**Claim.** The `consistent` definition (forcing 57.21, spacing 1/241) holds the values the dataset satisfies.

**Evidence.** On cells of constant `Kcoeff` the reference has $|r| = 0.0147$ normalised ($2.6\times10^{-4}$ in physical units), at float32 rounding level: consistent there. On the 16 % of cells near jumps it is 165.9. Two independent causes, both from F3:

1. The coefficient. Both definitions differentiate `Kcoeff`. The data were generated from `coeff`. Solving the same discrete problem with `Kcoeff` changes the solution by 1.7 % on average and up to 5.1 %; an exact residual built on `Kcoeff` would therefore pull predictions towards a different pressure field, by an amount of the same order as the accuracy differences studied (0.0005 to 0.005).
2. The operator. Even with `coeff`, the reference satisfies the scheme on the node grid with a spline-interpolated coefficient; no cell-grid five-point stencil reproduces it (`data_check.json`, all variants at about 2.8).

**Verdict: SUPPORTED BUT LIMITED.** "Data-consistent" is accurate for the forcing and the spacing, and away from jumps. It is not a residual the data satisfy near jumps. The README's own limitation ("corrects the forcing and the spacing only") is correct; it did not identify the coefficient mismatch.

## F5. The discrete residual near discontinuities

The finite-difference residual of the training loss is not a measure of PDE violation near jumps: the exact solution scores 165.9 there (normalised), and the trained models score below it (41.5 at `consistent` weight 0.1). The README states that "a lower mean residual is not the same as agreement with the reference" and that a model "can score below the reference by smoothing". Those statements are correct. The statement that the mean residual is a valid diagnostic of physical consistency at jumps would be **INCORRECT**; the project does not make it.

The exact residual of F3 is a valid diagnostic everywhere. Applied to the models (F8), it confirms that cell-grid and exact residuals rank the models in a similar order away from jumps.

## F6. Accuracy of `consistent` at weight 0.1

**Claim.** Similar prediction error (0.0475 against 0.0480); not evidence of a reproducible gain.

**Evidence.** Per-sample relative L2 errors from `eval_metrics.json`, paired over the 256 shared test samples (bootstrap, 10,000 resamples, seed 0):

| Model minus data-only | Mean difference | 95 % interval | Samples improved |
|---|---|---|---|
| `shipped` 0.001 | -0.00007 | [-0.00021, +0.00007] | 59 % |
| `shipped` 0.01 | +0.00077 | [+0.00038, +0.00118] | 43 % |
| `shipped` 0.1 | +0.00407 | [+0.00338, +0.00479] | 20 % |
| `shipped` 1 | +0.04985 | [+0.04797, +0.05162] | 0 % |
| `consistent` 0.001 | -0.00013 | [-0.00023, -0.00003] | 69 % |
| `consistent` 0.01 | -0.00020 | [-0.00055, +0.00016] | 61 % |
| `consistent` 0.1 | -0.00048 | [-0.00128, +0.00036] | 58 % |
| `consistent` 1 | +0.00568 | [+0.00385, +0.00760] | 39 % |

These intervals describe the variability over test samples for fixed trained models. They say nothing about variability over training seeds, which is unmeasured.

**Verdict: VERIFIED** that the error is the same within the test-sample resolution; **UNVERIFIED** that it is lower. The README's wording is correct. A claim of improved accuracy from physics weighting is not supported. (At weight 0.001 the interval excludes zero, but a 0.3 % relative change on one seed is not evidence of an effect.)

**Update after G1 (F14).** Over seeds 0, 1, 2 the paired difference at weight 0.1 is -0.00043, 95 % t interval over seeds [-0.0070, +0.0061], lower in two seeds and higher in one. The verdict stands: no detectable difference, improvement UNVERIFIED. The test-sample intervals of single seeds exclude zero in opposite directions for seeds 1 and 2, which confirms that they do not measure the effect of the setting.

## F7. The shipped residual degrades accuracy

**Evidence.** Table of F6: from weight 0.01 the error rises with intervals excluding zero; at weight 1 every test sample is worse and the error doubles. The mechanism is algebraically clear: the shipped residual of the exact solution is 56.7 on every constant-permeability cell, so the PDE term and the data term have different minimisers; the signed mean residual of the shipped models moves to -7.8, -17.4, -31.2 (consistent definition) as the weight grows, against ±0.5 for the consistent models.

**Verdict: SUPPORTED BUT LIMITED.** Established for seed 0 and 15 epochs. The size at weight 1 (+0.050 against a per-sample spread of 0.019) makes it very likely to survive more seeds; the 9 % at the upstream default 0.1 needs G1 and G2 of `EXPERIMENT_GAPS.md`.

**Update after G1 (F14).** At weight 0.1 the error is higher than data-only in all three seeds (+0.0041, +0.0022, +0.0056; mean +0.0040, +8.1 %), with a 95 % t interval over seeds of [-0.0002, +0.0081], t-test p = 0.054. The direction is reproduced; with three seeds the size is not established at the 5 % level. The upstream budget (G2) remains open.

## F8. Physics weighting lowers the residual (exact residual, new)

`scripts/evaluate_solver_residual.py` evaluates the existing checkpoints (inference only) under the exact residual of F3, with `coeff`, on interior nodes at least 6 nodes from the boundary (the crop-and-extend procedure is exact to rounding beyond 4 nodes, `reference_check.json`). Physical units, forcing 1. Results in `results/audit/local_solver_residual.json`.

| Model | Test relative L2 | Mean abs residual | Interface nodes | Other nodes |
|---|---|---|---|---|
| reference | 0 | 4.1e-04 | 4.7e-04 | 4.1e-04 |
| data only | 0.0480 | 2.840 | 12.16 | 2.535 |
| `shipped` 0.001 | 0.0479 | 1.886 | 11.23 | 1.580 |
| `shipped` 0.01 | 0.0488 | 1.066 | 9.65 | 0.785 |
| `shipped` 0.1 | 0.0521 | 0.896 | 8.75 | 0.639 |
| `shipped` 1 | 0.0978 | 1.005 | 8.14 | 0.771 |
| `consistent` 0.001 | 0.0479 | 1.777 | 11.15 | 1.470 |
| `consistent` 0.01 | 0.0478 | 0.946 | 9.71 | 0.659 |
| `consistent` 0.1 | 0.0475 | 0.565 | 8.30 | 0.312 |
| `consistent` 1 | 0.0537 | 0.425 | 7.42 | 0.196 |

The recomputed test errors equal the stored ones to 1.0e-07 relative.

**Verdict: VERIFIED for seed 0.** The residual reduction reported with the cell-grid residual also appears under the exact operator, for both definitions, so it is not an artefact of the stencil. For `consistent` it decreases monotonically with the weight; for `shipped` it rises again from weight 0.1 (0.896) to 1 (1.005), as expected from a loss that targets the wrong forcing. All models remain three to four orders of magnitude above the reference, most of all at interface nodes, where even the best model is about 16,000 times above it.

**Update after G1 (F14).** At weight 0.1 the reduction reproduces in every seed for both definitions and both residuals.

## F9. Residual against accuracy, by region (new)

Relative L2 error restricted to cells with locally constant `Kcoeff` and to the others (border of 2 excluded), same file:

| Model | Constant-`Kcoeff` cells | Other cells | All cells |
|---|---|---|---|
| data only | 0.0434 | 0.0623 | 0.0480 |
| `shipped` 0.001 | 0.0434 | 0.0622 | 0.0479 |
| `shipped` 0.01 | 0.0439 | 0.0637 | 0.0488 |
| `shipped` 0.1 | 0.0459 | 0.0660 | 0.0521 |
| `shipped` 1 | 0.0912 | 0.0860 | 0.0978 |
| `consistent` 0.001 | 0.0433 | 0.0621 | 0.0479 |
| `consistent` 0.01 | 0.0431 | 0.0627 | 0.0478 |
| `consistent` 0.1 | 0.0423 | 0.0645 | 0.0475 |
| `consistent` 1 | 0.0491 | 0.0646 | 0.0537 |

At `consistent` weight 0.1, the error falls on constant-permeability cells and rises near jumps; the overall error is unchanged. This is consistent with the README's reading (smoothing near jumps) and with F4: near jumps the PDE term enforces an operator and a coefficient that the data do not satisfy. **VERIFIED** as a measurement for one seed; the attribution to F4 is **SUPPORTED BUT LIMITED** (G3 and G4 would test it).

**Update after G1 (F14).** The higher error near jumps at `consistent` 0.1 reproduces in all three seeds (+0.0022, +0.0020, +0.0038; interval [+0.0003, +0.0051]). The lower error on constant-permeability cells does not: -0.0011, -0.0044, +0.0017 (interval [-0.0088, +0.0063]).

## F10. Data splitting and leakage

- Index ranges 0 to 101, 102 to 203, 204 to 459 are disjoint (`data.split_bounds`, `tests/test_data.py`, `manifest.json`). **VERIFIED.**
- No two of the 460 `coeff` fields are identical (SHA-1 of each field). **VERIFIED.**
- Similar fields exist: the closest test-training pair agrees on 98.5 % of cells and the solutions differ by 1.75 % relative L2; 40 of 256 test samples agree with some training sample on more than 95 % of cells. This is what a smooth random-field prior produces and not leakage, but the test set is not out of distribution.
- The test set is not used for training, model selection or early stopping (final-epoch checkpoints; the weight grid was chosen from training-loss magnitudes). **VERIFIED** from `engine.py` and `study.py`.
- The scaling constants equal the standard deviations of `coeff` (4.49995) and `sol` (0.0038946 over the 460 samples) to 0.3 %. They are fixed upstream constants and only rescale; over which samples upstream computed them is **UNVERIFIED**, without practical consequence.

## F11. Recomputation of the tracked summaries

From `eval_metrics.json` (256 per-sample errors), `train_metrics.json` and `history.csv` of all 9 runs: mean and standard deviation of the test error, final data, PDE and total losses, total = data + coefficient x PDE, residual entries of `summary.csv` and `summary.json`, and `by_setting` all agree to better than $10^{-6}$ relative. **VERIFIED.**

The `median` entries differ from the conventional median by up to 0.5 %: `metrics.summarise` uses `torch.median`, which returns the lower of the two middle values for an even count. The tracked numbers are correct for that definition. The docstring now states it; the behaviour and the historical files are unchanged. The README does not quote medians.

`results/local/` is a byte-identical copy of `runs/local/` without checkpoints. The checkpoints reproduce the stored test error (F8 script). Training ran at commit `d1bd2ca`, evaluation at `9ebcaab`, both clean; the code between them changed only plotting and `plot_samples`.

## F12. Implementation against upstream

Architecture, scaling, crop, splits, PDE class, finite-difference module, border, L1 reduction, loss expression, optimiser and per-sample schedule are identical to upstream (detail and evidence in `UPSTREAM_COMPARISON.md`). The behavioural changes (seeding, residual on CPU under MPS, 15-epoch compressed schedule in `local`, added test split and metrics) are all stated in the README. **VERIFIED.**

## F13. Other README statements

| Statement | Check | Verdict |
|---|---|---|
| 2,365,169 parameters | `train_metrics.json` | VERIFIED |
| Error +9 % at 0.1, x2.0 at 1 (`shipped`); +12 % at 1 (`consistent`) | 0.0521/0.0480, 0.0978/0.0480, 0.0537/0.0480 | VERIFIED |
| Residual lowered by a factor of 9 at `consistent` 0.1 | 147.5/16.1 = 9.2 | VERIFIED |
| Data-only residual 5.7 times the reference | 147.5/26.0 | VERIFIED |
| Weighted PDE term "a tenth of the data term at 0.001" | 0.079 to 0.099 | VERIFIED |
| Weighted PDE term "two to three times the data term at 1" | 2.27 (`consistent`), 3.38 (`shipped`) | INCORRECT for `shipped`; corrected in the README to "2.3 to 3.4". Also at the upstream default (`shipped`, 0.1) the weighted PDE term is already 2.2 times the data term |
| Validation error still falling at epoch 15 | minimum at the last epoch in 8 of 9 runs (`shipped` 1: epoch 14) | VERIFIED |
| 67 minutes of training, 171 to 238 ms per step | 66.7 min, 171 to 238 ms | VERIFIED |

## F14. Seed study at weight 0.1 (addendum, G1)

**Design.** G1 exactly as specified in `EXPERIMENT_GAPS.md`: data-only, `shipped` 0.1 and `consistent` 0.1 trained with seeds 1 and 2, compared with the seed-0 runs of the reduced study. Command:

```bash
python scripts/train.py --config local name=local_seeds "study.seeds=[1,2]" "study.residuals=[shipped,consistent]" "study.physics_weights=[0,0.1]" evaluation.plot_samples=3
```

`evaluation.plot_samples=3` only selects how many test samples are stored in `fields.npz`; it is the value the seed-0 study had at training time. An earlier attempt with the same configuration at 15:07 UTC stopped before its first run finished and wrote no run records; its directory is kept as `runs/local_seeds_aborted_20261024T150752Z/` (untracked) and was not used.

**Provenance.** Trained at commit `4c18b90` (clean tree) on Apple M3 Pro, 18 GB, macOS 26.6.2, PyTorch 2.14.1, PhysicsNeMo commit `b45a5c8`, FNO on MPS and residual on CPU, as for seed 0 (commit `d1bd2ca`, which differs from `4c18b90` in plotting, stored evaluation fields, the reference solver and the lock; `engine.py`, `physics.py`, `model.py`, `data.py` and the training settings are unchanged). `results/local_seeds/run_manifest.json` lists for all nine runs the hash of the training settings (identical: `cc5a3158...`; excluded only `name`, `output_dir`, `study`, `evaluation` and the unused top-level `physics_weight`), a per-run configuration hash, dataset file hashes (identical), and SHA-256 of both checkpoint files. The training took 1063.5 s for the six runs (148 to 203 s each, 64 to 90 ms per step); evaluation of the summaries and of the exact residual added about 3 minutes. The stored test errors are recomputed from the checkpoints by `scripts/evaluate_solver_residual.py` to better than 1e-5 relative.

**Method.** `scripts/seed_study.py`. For each seed, the model is compared with the data-only model of the same seed on the same 256 test samples. Errors: paired differences; residuals: log ratios. Mean over seeds with a Student-t 95 % interval (2 degrees of freedom), t-test p-value and exact sign test (smallest attainable two-sided p with three seeds: 0.25). Per seed, a test-sample bootstrap (10,000 resamples) of the error difference, as in F6. No correction for multiple comparisons; the primary comparison is the overall relative L2 error of `consistent` 0.1 against data-only.

**Results** (per-seed values in the README and `results/audit/seed_study.json`).

| Model minus data-only | Overall error | Constant-`Kcoeff` cells | Near jumps | Exact residual (ratio) | Exact residual, interface nodes | Training residual (ratio) |
|---|---|---|---|---|---|---|
| `shipped` 0.1 | +0.0040 [-0.0002, +0.0081], p = 0.054 | +0.0022 [-0.0022, +0.0066] | +0.0042 [+0.0022, +0.0062] | 0.30 [0.10, 0.92] | 0.69 [0.56, 0.84] | 0.23 [0.06, 0.85] |
| `consistent` 0.1 | -0.0004 [-0.0070, +0.0061], p = 0.80 | -0.0013 [-0.0088, +0.0063] | +0.0027 [+0.0003, +0.0051] | 0.19 [0.06, 0.62] | 0.66 [0.54, 0.82] | 0.11 [0.03, 0.41] |

Seed spread (mean ± standard deviation over seeds): overall error 0.0489 ± 0.0013 (data-only), 0.0529 ± 0.0009 (`shipped`), 0.0485 ± 0.0018 (`consistent`); exact residual 3.22 ± 1.49, 0.894 ± 0.004, 0.584 ± 0.017. Per-seed test-sample bootstrap of `consistent` minus data-only: seed 0 -0.0005 [-0.0013, +0.0004], seed 1 -0.0031 [-0.0041, -0.0020], seed 2 +0.0022 [+0.0013, +0.0032].

**Interpretation.**

1. *Residual.* The reduction is reproducible in direction for both definitions, under the smoothed-coefficient training residual and under the exact operator: every seed, both settings. Its size is not a property of the physics term alone: the weighted models land at nearly the same exact residual in every seed, while the data-only residual varies by a factor of 2.5, so the ratio is set mostly by how rough the data-only model happens to be. At interface nodes the exact residual falls by only a third, which is consistent with F4 (the training residual enforces a coefficient and an operator that the data do not satisfy at jumps). The ratio intervals exclude 1 but are wide; with three seeds they rest on a t distribution with 2 degrees of freedom.
2. *Accuracy, `consistent`.* No effect detected. The mean change (-0.9 %) is smaller than the seed-to-seed standard deviation of the data-only error (2.7 %), the sign differs between seeds, and the interval covers -14 % to +13 % of the error. A practically relevant improvement or degradation of that size cannot be excluded with three seeds; an improvement is not supported. The reduced residual does not imply a reduced error.
3. *Accuracy, `shipped`.* Worse in every seed and by more than the data-only seed spread in two of three (+8 % on average). Statistically borderline at n = 3 (p = 0.054; the sign test cannot reach significance with three seeds). This agrees in direction with F7 and with the algebraic mechanism there.
4. *Regions.* Near jumps both weighted settings are worse in every seed with intervals excluding zero (nominal p = 0.04 and 0.01, uncorrected for the number of comparisons). Away from jumps no consistent effect. The seed-0 picture of F9 (better away from jumps, worse near them, unchanged overall) is half reproduced: the degradation near jumps is, the improvement away from them is not.
5. *Single-seed evidence.* The test-sample intervals of seeds 1 and 2 each exclude zero, in opposite directions. Any accuracy conclusion from one seed with test-sample intervals, including the weight-0.001 interval of F6, should be read as a property of one trained pair, not of the setting.

**Verdict.** Residual reduction at weight 0.1: **VERIFIED** over three seeds (direction; size depends on the data-only seed). Accuracy improvement from `consistent` weighting: **UNVERIFIED**, no detectable effect. Accuracy degradation from `shipped` at 0.1: **SUPPORTED BUT LIMITED**, consistent direction, interval just includes zero. All at 15 epochs; G2 is open.

## F15. The upstream training budget (G2, addendum 2026-11-07)

**Design.** Pre-registered in [`FULL_BUDGET_PLAN.md`](FULL_BUDGET_PLAN.md) (committed in `eaca64c` before any run): 50 epochs of 102 steps (5100 optimiser steps, the upstream budget and per-step learning-rate schedule), batch size 1, seeds 0, 1, 2. Stage 1 (`configs/full_mps.yaml`): data only, `shipped` 0.1, `consistent` 0.1. Stage 2 (`configs/full_mps_sweep.yaml`): both residuals at 0.001, 0.01, 1. Primary comparison: overall test relative L2, `consistent` 0.1 minus data only, paired by seed, Student-t 95 % interval over seeds; everything else secondary, with Holm adjustment over the seven secondary settings.

**Hardware, not CUDA.** All 27 runs trained on Apple MPS (M3 Pro, 18 GB; FNO on MPS, cell-grid residual on CPU as in every earlier study), not on the CUDA GPU the plan of `kaggle/run_cuda.ipynb` targets. A CUDA replication has not been run. MPS training is bitwise repeatable for the same seed (`results/audit/determinism.json`: two one-epoch runs per setting give identical parameters) and differs from CPU at about 1e-2 in the parameters after one epoch.

**Provenance.** Trained from a clean git worktree at `eaca64c`. `study_metadata.json` of both studies records `git_dirty: true`; the only changes in that worktree were untracked audit outputs written by the same queue (`results/audit/determinism.json` and the stage-1 audit files), and `git diff HEAD` there is empty. The first execution session ended while stage 2 was running: 15 of 18 runs had finished with records; the run in progress (`consistent_w0.001_seed2`) left an empty directory and was retrained from scratch with `scripts/train.py --resume`, which skips runs that have `train_metrics.json`. Every run has its configuration, seed, 5100 steps, a 50-epoch history, the final checkpoint (not tracked) and test metrics; `scripts/full_study.py` asserts that the stored test error equals the error recomputed from the checkpoint by `scripts/evaluate_solver_residual.py` to 1e-5 relative, for every run.

**Results** (`results/audit/full_study.json`, `figures/full_*.png`). Means over seeds 0 to 2; differences paired by seed with 95 % t intervals (df = 2).

| Setting | Test rel. L2 | SD over seeds | Constant-`Kcoeff` cells | Near jumps | Exact residual | Difference to data only | Reading (Holm p) |
|---|---|---|---|---|---|---|---|
| data only | 0.0377 | 0.0014 | 0.0344 | 0.0489 | 2.43 | | |
| `consistent` 0.001 | 0.0367 | 0.0012 | 0.0334 | 0.0476 | 1.46 | -2.8 % [-5.4, -0.2] | improvement unadjusted (0.17) |
| `consistent` 0.01 | 0.0334 | 0.0012 | 0.0301 | 0.0442 | 0.78 | -11.5 % [-14.1, -9.0] | improvement (0.016) |
| **`consistent` 0.1 (primary)** | **0.0295** | 0.0012 | 0.0259 | 0.0414 | 0.49 | **-21.7 % [-27.1, -16.3]** | **improvement (p = 0.003)** |
| `consistent` 1 | 0.0348 | 0.0009 | 0.0304 | 0.0483 | 0.40 | -7.7 % [-18.7, +3.4] | inconclusive (0.27) |
| `shipped` 0.001 | 0.0367 | 0.0009 | 0.0333 | 0.0477 | 1.58 | -2.8 % [-6.8, +1.1] | inconclusive (0.27) |
| `shipped` 0.01 | 0.0350 | 0.0011 | 0.0317 | 0.0451 | 1.03 | -7.3 % [-10.1, -4.4] | improvement (0.042) |
| `shipped` 0.1 | 0.0372 | 0.0004 | 0.0332 | 0.0407 | 0.90 | -1.4 % [-8.8, +5.9] | inconclusive (0.49) |
| `shipped` 1 | 0.0913 | 0.0006 | 0.0840 | 0.0768 | 1.03 | +142 % [+129, +155] | degradation (0.003) |

Primary comparison: -0.0082 [-0.0102, -0.0061], lower in all three seeds (-0.0085, -0.0088, -0.0073); the interval excludes zero and lies entirely below -5 % of the data-only error. Per-seed test-sample bootstrap intervals of the same difference: [-0.0096, -0.0075], [-0.0101, -0.0076], [-0.0084, -0.0061]; they agree with the seed interval here, unlike at 15 epochs (F14). Near jumps: -0.0075 [-0.0084, -0.0065]; on constant-`Kcoeff` cells -0.0085 [-0.0108, -0.0062]. Exact residual ratio 0.20 [0.16, 0.25], at interface nodes 0.67 [0.65, 0.69]; training (cell-grid) residual ratio 0.086 [0.071, 0.104].

Stopping rule: the half-width of the primary interval after stage 1 (2.04e-3) exceeds 5 % of the data-only mean (1.89e-3), so the pre-registered stage 3 (seeds 3 and 4 of stage 1) was triggered and queued; its result is reported separately for n = 5 when it exists, and the n = 3 result above is the one the stopping rule was applied to.

**Interpretation.**

1. *Budget changes the accuracy conclusion.* At 15 epochs `consistent` 0.1 had no detectable effect (-0.9 %, F14); at 50 epochs it lowers the error by about a fifth in every seed, by many times the seed spread (the mean difference is 5.7 data-only seed standard deviations). Between 15 and 50 epochs the data-only error falls from 0.049 to 0.038 and the `consistent` 0.1 error from 0.048 to 0.030, for the same seeds.
2. *The jump-region degradation of F9 and F14 does not persist.* At 15 epochs both weighted settings were worse near jumps in every seed; at 50 epochs both are better there in every seed (`consistent` -0.0075, `shipped` -0.0082). The near-jump degradation was a property of the shorter training, not of the residual.
3. *Residual and accuracy separate.* With increasing `consistent` weight the exact residual falls monotonically (ratios 0.60, 0.32, 0.20, 0.17 at 0.001 to 1), while the error falls up to 0.1 and rises again at 1, where the interval includes zero. The trained `consistent` 0.1 models have a lower cell-grid residual (test mean 10.6) than the reference pressure itself (23 to 27; `results/audit/solver_operator_check.json` and the summary's `reference_residual`): the training residual is minimised by fields that differ from the reference, so beyond some weight a lower residual moves the model away from the data.
4. *The corrected forcing matters.* At equal weight `consistent` is more accurate than `shipped` from 0.01 upwards, and `shipped` at weight 1 more than doubles the error (as at 15 epochs, F7), while its exact residual stops falling after 0.1. `shipped` 0.01 is nevertheless better than data only: at small weight its main effect is not the wrong forcing.
5. *A plausible mechanism, not tested.* Data only fits the training set much more closely (training relative L2 0.0083) than `consistent` 0.1 (0.0177) and has the larger test error; its training error is 4.5 times its test error, against 1.7 times for `consistent` 0.1. The residual acts as a regulariser that trades training fit for generalisation. This is consistent with the numbers and is not separately tested (no other regulariser was compared).
6. *Convergence.* Over the last 10 % of the steps the data-only validation error changes by -0.6 % to +0.1 %; `consistent` 0.1 still falls by 1.4 % to 4.3 %. The improvement is therefore not an artefact of an unconverged data-only model.
7. *Cost.* 65 ms per step for data only and 71 ms with either cell-grid residual on MPS (8 % more), 515 to 533 s per run; these timings share the machine with other work and are indicative only. Inference cost does not depend on the training loss.

**Verdict.** At the upstream budget, on MPS, with three seeds: `consistent` 0.1 improves the test error over data only, **VERIFIED** for this setting under the pre-registered rule (seeds 0 to 2; stage 3 pending at the time of writing). The improvement depends on the weight (largest of the four weights tested at 0.1; inconclusive at 1). `shipped` 1 degrades accuracy: **VERIFIED**. The near-jump degradation of F9 and F14 is **not reproduced** at 50 epochs. Generality beyond this dataset, architecture, optimiser and these weights is not tested; tuning parity is not addressed (all settings use the upstream learning rate and schedule).

## F16. Coefficient and operator of the residual (G3, G4, addendum 2026-11-07)

**Design.** Pre-registered in [`OPERATOR_AND_LABELS_PLAN.md`](OPERATOR_AND_LABELS_PLAN.md), committed with the configurations in `beaac7f` before any of these runs. A 2 x 2 factorial at weight 0.1 and the G2 budget (5100 steps, seeds 0 to 2): operator (cell-grid finite differences, or the generator's own operator `solve_gwf.m` on nodes) crossed with coefficient (`Kcoeff`, the network input, or `coeff`, the field the generator solved with). The cell-grid/`Kcoeff` cell is `consistent` 0.1 of F15 (same seeds, budget and code path); the other three are `consistent_coeff`, `solver_kcoeff` and `solver` (`configs/full_mps_operator.yaml`), plus `solver` at weight 1 (`configs/full_mps_solver_w1.yaml`). Primary (G4): overall test error, `solver` 0.1 minus data only. Everything else secondary, with Holm over the nine G3/G4 comparisons.

**Provenance.** Trained on MPS from a worktree at `beaac7f`: `full_mps_operator` from a clean tree (`git_dirty: false`), `full_mps_solver_w1` with `git_dirty: true` caused only by the queue's own untracked audit outputs (`git diff HEAD` empty). 9 of 9 and 3 of 3 runs complete with 5100 steps and a 50-epoch history; stored and recomputed test errors agree for every run. Training-split errors from `scripts/evaluate_generalisation.py`. The queue waited about 12 hours for a machine-wide GPU lock held by another project; this delayed but did not alter any run.

**Operator checks on the real fields** (`scripts/check_solver_operator.py`, `results/audit/solver_operator_check.json`, CPU, first 16 training and test samples). The torch `solver` residual equals the NumPy re-implementation of `solve_gwf.m` exactly in float64 (maximum difference 0.0). The stored reference pressure, evaluated exactly as in training (float32, normalised units), leaves a mean |r| of 0.13 % of the forcing under `solver`, with no concentration at jumps (0.15 % there), against 41 to 50 % under `consistent` and `solver_kcoeff` and 11 % under `consistent_coeff`, almost all of it near jumps (near-jump |r| 3.6 to 10.9 times the forcing). The `solver` loss of the reference is 0.065 against 23 to 25 for `consistent`. Autograd gradients of the `solver` loss agree with central differences: to 4e-13 for a quadratic loss of the same residual, and to 9e-4 for the L1 training loss at eps = 1e-6, where the remaining difference comes from residual nodes whose sign changes within the step (kinks of |r|). So `solver` is the operator for which the reference pressure is (to float32 precision) a zero of the training residual; the other three are not, mainly near jumps.

**Results** (`results/audit/full_study.json`, `figures/full_operator.png`). Means over seeds 0 to 2; relative differences to data only (0.0377) with paired 95 % t intervals.

| Setting | Operator | Coefficient | Test rel. L2 | SD over seeds | Train rel. L2 | Near jumps | Exact residual | Difference to data only | Near-jump difference | Reading (Holm p) |
|---|---|---|---|---|---|---|---|---|---|---|
| data only | | | 0.0377 | 0.0014 | 0.0083 | 0.0489 | 2.43 | | | |
| `consistent` 0.1 | cell grid | `Kcoeff` | 0.0295 | 0.0012 | 0.0177 | 0.0414 | 0.49 | -21.7 % [-27.1, -16.3] | -15.3 % [-17.3, -13.4] | improvement (G2 primary) |
| `consistent_coeff` 0.1 | cell grid | `coeff` | 0.0309 | 0.0015 | 0.0198 | 0.0455 | 0.43 | -18.0 % [-26.0, -10.0] | -6.9 % [-15.1, +1.4] | improvement unadjusted (0.083) |
| `solver_kcoeff` 0.1 | generator | `Kcoeff` | 0.0295 | 0.0013 | 0.0174 | 0.0412 | 0.49 | -21.7 % [-26.4, -17.0] | -15.7 % [-17.6, -13.9] | improvement (0.022) |
| **`solver` 0.1 (G4 primary)** | generator | `coeff` | **0.0304** | 0.0017 | 0.0187 | 0.0438 | 0.43 | **-19.4 % [-26.8, -12.0]** | -10.4 % [-20.8, +0.1] | **improvement (p = 0.008)** |
| `solver` 1 | generator | `coeff` | 0.0370 | 0.0025 | 0.0335 | 0.0495 | 0.29 | -1.9 % [-24.0, +20.1] | +1.3 % [-24.8, +27.4] | inconclusive (1.00) |

G4 primary: -0.0073 [-0.0101, -0.0045], lower in all three seeds (-0.0082, -0.0077, -0.0061). Exact-residual ratio to data only 0.18 [0.14, 0.23].

Factorial contrasts on the overall error (paired, df = 2, unadjusted): coefficient (`coeff` minus `Kcoeff`) +0.0014 [+0.0003, +0.0025] at the cell-grid operator and +0.0009 [-0.0004, +0.0021] at the generator operator; operator (generator minus cell grid) +0.0000 [-0.0003, +0.0003] with `Kcoeff` and -0.0005 [-0.0012, +0.0002] with `coeff`. Near jumps the coefficient effect is +0.0041 [+0.0011, +0.0072] (+10 %) at the cell-grid operator and +0.0026 [-0.0016, +0.0068] at the generator operator. `solver` against `consistent` directly: +2.9 % [-1.3, +7.1], inconclusive (Holm 0.40). `consistent_coeff` against `consistent`: +4.7 % [+1.1, +8.3], a degradation by the unadjusted interval that is not significant after Holm (0.21). `solver` 1 against `solver` 0.1: +21.7 % [+1.0, +42.4], higher in all three seeds, a degradation by the unadjusted interval (Holm 0.28); its exact residual is lower by a factor 0.67 [0.52, 0.85].

**Pre-stated mechanistic prediction.** The plan predicted that, if the near-jump degradation of `consistent` (F9, F14) were caused by the coefficient and operator mismatch of F4, the near-jump difference to data only would be positive for `consistent` and not positive for `solver`. At this budget the premise does not hold: `consistent` 0.1 is *better* near jumps than data only in every seed (F15), so there is no near-jump degradation for the mismatch to explain. `solver` is also not worse near jumps (all three seeds lower; interval [-20.8, +0.1] %). The prediction is therefore not testable as stated at 50 epochs; at 15 epochs, where the degradation was observed, the operator variants were not run.

**Interpretation.**

1. *A correct operator is not what makes the residual help.* The four residuals differ widely in how far the reference pressure is from satisfying them (mean |r| from 0.13 % to 50 % of the forcing), yet all four improve the error over data only by 18 to 22 %, and the operator makes no material difference with `Kcoeff` (`solver_kcoeff` minus `consistent` +0.0 % [-1.1, +1.1]). The benefit at this budget does not depend on the residual being exact for the reference solution.
2. *Using the generator's coefficient does not help and may hurt slightly.* Replacing `Kcoeff` by `coeff` lowers the exact residual of the trained models by about 13 % (ratio 0.87 to 0.88) but raises the error, by an amount that is unadjusted-significant at the cell-grid operator and mostly near jumps. This is a second instance of the residual-accuracy separation of F15: a residual that is lower on the generator's own operator does not give a more accurate field on the test metric, which measures the error against the stored pressure on the network's cell grid.
3. *A possible explanation, not tested.* The network receives `Kcoeff`; a residual built on `coeff`, which is sharper near jumps than the input it sees, asks the network to reproduce structure near jumps that is not in its input. This would explain why the coefficient effect is concentrated near jumps, and is offered as a hypothesis only.
4. *Weight 1 repeats the G2 pattern with the exact operator.* `solver` 1 lowers the exact residual to 0.12 [0.07, 0.19] of data only, the lowest of all settings, and gains nothing in accuracy (-1.9 %, interval from -24 % to +20 %), as `consistent` 1 did in F15. Its training error (0.033) is close to its test error (0.037): at this weight the residual term dominates and the model no longer fits the labelled data closely. A residual that the reference satisfies exactly does not remove the trade-off between residual and data fit.
5. *Cost.* 66 to 81 ms per optimiser step on MPS for the four residuals, against 65 to 71 ms for data only and the cell-grid residuals in F15; the generator operator is evaluated on CPU. Timings share the machine with other work and are indicative only.

**Verdict.** `solver` 0.1 improves the test error over data only at the upstream budget, on MPS, with three seeds: **VERIFIED** (G4 primary). That the solver-consistent residual is *better* than the corrected cell-grid residual: **not supported** (inconclusive, point estimate slightly worse). That the coefficient mismatch explains the jump-region degradation: **not testable at this budget**, because the degradation does not occur at 50 epochs. `solver` 1 has no detectable effect over data only (inconclusive) and is worse than `solver` 0.1 by the unadjusted interval: the benefit depends on the weight for the exact operator as for the cell-grid one.

## Software checks

- `python -m pytest -q`: 48 passed (41 existing, 7 new), offline and on CPU. After the addendum: 59 passed (accelerator lock, paired statistics, and the agreement of `results/audit/seed_study.json` with the per-run records).
- `ruff check --select E,F,W`: clean. `--select B` adds nine `B905` (`zip` without `strict=`), stylistic only.
- No implementation error that changes a tracked number was found.

## Changes made in the audit

- `src/pino_darcy/reference.py`: re-implementation of the data-generation solver and its exact residual; `tests/test_reference.py`.
- `scripts/verify_reference.py`, `scripts/evaluate_solver_residual.py`: new verification; outputs in `results/audit/`.
- `metrics.summarise`: docstring states the lower-median convention (no behaviour change).
- `requirements.lock`: exact environment.
- README: the corrected ratio (F13), the coefficient finding (F3, F4), test count, links to these documents.
- Not changed: `results/local/`, `results/data_check.json`, `figures/`, training and evaluation code.
- Addendum (F14): `src/pino_darcy/gpulock.py` and its use in `scripts/train.py`, `scripts/evaluate.py`, `scripts/evaluate_solver_residual.py` (a machine-wide accelerator lock; no change to training or evaluation); `metrics.paired_seed_interval`, `metrics.paired_bootstrap`; `scripts/seed_study.py`; new results `results/local_seeds/`, `results/audit/local_seeds_solver_residual.json`, `results/audit/seed_study.json`, `figures/seed_study.png`; tests `tests/test_gpulock.py`, `tests/test_seed_study.py`. Existing results and figures unchanged.

## Claims the evidence supports

Safe to state, with the qualifiers given:

1. The upstream PINO Darcy example at commit `b45a5c8` uses a forcing constant that is the reciprocal of the value implied by its own normalisation (0.0175 instead of 57.2), and a grid spacing of 1/240 where the stored fields have 1/241. Both are verified from the data and from the original solver.
2. The reference pressure of `Darcy_241` is reproduced to float32 rounding by the original FNO solver applied to the two-valued coefficient; the smoothed field used as model input and in the residual is not the coefficient of that equation.
3. In a reduced, single-seed study (15 epochs), training with the shipped residual increased the test error at weights of 0.01 and above (doubling it at weight 1), while the corrected residual reduced the PDE residual by about an order of magnitude at weight 0.1 with no detectable change in test error.
4. At weight 0.1 and 15 epochs, over three training seeds: both residual definitions lower the PDE residual in every seed, under the training residual and under the exact operator of the data generator; the corrected residual has no detectable effect on the test error (-0.0004, 95 % interval over seeds -0.0070 to +0.0061); the shipped residual raised the error in every seed (+8 % on average, interval just including zero).
5. A lower finite-difference residual did not imply closer agreement with the reference; near permeability jumps the weighted models had a higher error in every seed.

Not supported: that physics weighting improves accuracy; that any result holds at the upstream budget, or across seeds at weights other than 0.1; that the corrected residual is exact at jumps; any statement about the DeepONet variant or about data-scarce regimes.

## Recommendation

The project is scientifically defensible as a careful reproduction-and-diagnosis study of one upstream example, and its tracked results are reproducible from its records. It is not evidence for a performance claim. G1 has been run (F14): the residual finding is reproducible over seeds, an accuracy gain is not supported, and three seeds leave intervals of about ±14 % on the error. Next, in order: more seeds at weight 0.1 if a sharper accuracy statement is wanted (about 3 minutes per run locally under light load), G2 for the upstream budget, and G3 and G4, which test the coefficient and operator mismatch identified here and the jump-region degradation that F14 reproduces.
