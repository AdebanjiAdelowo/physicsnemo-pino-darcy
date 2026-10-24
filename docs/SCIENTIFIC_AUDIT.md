# Scientific and software audit

Audit date 2026-10-24. Object: this repository at commit `6ce3229` (the state before the audit), its tracked results (`results/data_check.json`, `results/local/`), and its relation to the NVIDIA PhysicsNeMo example `examples/cfd/darcy_physics_informed` at commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`. The upstream comparison is in [`UPSTREAM_COMPARISON.md`](UPSTREAM_COMPARISON.md), missing experiments in [`EXPERIMENT_GAPS.md`](EXPERIMENT_GAPS.md), the checklist in [`REPRODUCIBILITY_CHECKLIST.md`](REPRODUCIBILITY_CHECKLIST.md).

Classification:

- **VERIFIED**: reproduced in this audit from code, data or result files, or proven.
- **SUPPORTED BUT LIMITED**: consistent with the evidence, but the evidence cannot carry the general claim (one seed, reduced budget, test-sample variability only).
- **UNVERIFIED**: not checked, or not checkable with what exists.
- **INCORRECT**: contradicted by the evidence.

Everything measured in this audit is CPU or Apple MPS inference and NumPy/SciPy computation. No model was trained.

## Summary

The implementation reproduces the upstream example faithfully, the tracked numbers are what the code produced, and the README states its limits accurately. The audit confirms the two residual corrections and adds one finding the project had not identified: the reference pressure was generated from the two-valued coefficient `coeff` with a specific discrete scheme on a staggered node grid, while both residual definitions use the smoothed model input `Kcoeff` on the cell-centred grid. This explains the large residual of the reference near permeability jumps, it qualifies the name "data-consistent", and it gives an exact residual against which the trained models can be measured. Under that exact residual the physics-weighted models still have a lower residual than the data-only model, so the main residual finding stands; the accuracy finding remains a single-seed observation with no established improvement.

| ID | Claim or question | Verdict |
|---|---|---|
| F1 | Upstream forcing 0.01748 is wrong; the scaled equation needs 57.21 | VERIFIED |
| F2 | Stored fields are cell-centred with spacing 1/241, not 1/240 | VERIFIED |
| F3 | What the reference pressure solves | VERIFIED (new) |
| F4 | The `consistent` residual is consistent with the data | SUPPORTED BUT LIMITED: only away from jumps |
| F5 | The cell-grid residual is a valid diagnostic near jumps | INCORRECT as a measure of PDE violation there; the README already warns about this |
| F6 | `consistent`, weight 0.1, is as accurate as data-only (0.0475 against 0.0480) | VERIFIED as "no detectable difference"; any improvement UNVERIFIED |
| F7 | `shipped` raises the error from weight 0.01 upwards | SUPPORTED BUT LIMITED (one seed) |
| F8 | Physics weighting lowers the PDE residual | VERIFIED for this seed, also under the exact residual (new) |
| F9 | Lower residual is not closer agreement with the reference | VERIFIED; region-resolved error added (new) |
| F10 | Splits are disjoint, no leakage | VERIFIED |
| F11 | Tracked summaries match the per-run records | VERIFIED, with one labelling issue (median) |
| F12 | Implementation matches upstream | VERIFIED |
| F13 | Smaller README statements | one INCORRECT figure (corrected), others VERIFIED |

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

## F7. The shipped residual degrades accuracy

**Evidence.** Table of F6: from weight 0.01 the error rises with intervals excluding zero; at weight 1 every test sample is worse and the error doubles. The mechanism is algebraically clear: the shipped residual of the exact solution is 56.7 on every constant-permeability cell, so the PDE term and the data term have different minimisers; the signed mean residual of the shipped models moves to -7.8, -17.4, -31.2 (consistent definition) as the weight grows, against ±0.5 for the consistent models.

**Verdict: SUPPORTED BUT LIMITED.** Established for seed 0 and 15 epochs. The size at weight 1 (+0.050 against a per-sample spread of 0.019) makes it very likely to survive more seeds; the 9 % at the upstream default 0.1 needs G1 and G2 of `EXPERIMENT_GAPS.md`.

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

## Software checks

- `python -m pytest -q`: 48 passed (41 existing, 7 new), offline and on CPU.
- `ruff check --select E,F,W`: clean. `--select B` adds nine `B905` (`zip` without `strict=`), stylistic only.
- No implementation error that changes a tracked number was found.

## Changes made in the audit

- `src/pino_darcy/reference.py`: re-implementation of the data-generation solver and its exact residual; `tests/test_reference.py`.
- `scripts/verify_reference.py`, `scripts/evaluate_solver_residual.py`: new verification; outputs in `results/audit/`.
- `metrics.summarise`: docstring states the lower-median convention (no behaviour change).
- `requirements.lock`: exact environment.
- README: the corrected ratio (F13), the coefficient finding (F3, F4), test count, links to these documents.
- Not changed: `results/local/`, `results/data_check.json`, `figures/`, training and evaluation code.

## Claims the evidence supports

Safe to state, with the qualifiers given:

1. The upstream PINO Darcy example at commit `b45a5c8` uses a forcing constant that is the reciprocal of the value implied by its own normalisation (0.0175 instead of 57.2), and a grid spacing of 1/240 where the stored fields have 1/241. Both are verified from the data and from the original solver.
2. The reference pressure of `Darcy_241` is reproduced to float32 rounding by the original FNO solver applied to the two-valued coefficient; the smoothed field used as model input and in the residual is not the coefficient of that equation.
3. In a reduced, single-seed study (15 epochs), training with the shipped residual increased the test error at weights of 0.01 and above (doubling it at weight 1), while the corrected residual reduced the PDE residual by about an order of magnitude at weight 0.1 with no detectable change in test error.
4. A lower finite-difference residual did not imply closer agreement with the reference; near permeability jumps the corrected residual traded accuracy for residual.

Not supported: that physics weighting improves accuracy; that any result holds at the upstream budget or across seeds; that the corrected residual is exact at jumps; any statement about the DeepONet variant or about data-scarce regimes.

## Recommendation

The project is scientifically defensible as a careful reproduction-and-diagnosis study of one upstream example, and its tracked results are reproducible from its records. It is not yet evidence for a performance claim. Before any accuracy statement, run G1 (seeds, about 45 minutes locally); for a stronger contribution, implement G3 and G4, which test the coefficient and operator mismatch identified here.
