# Comparison with the NVIDIA upstream example

Upstream: [NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo), commit `b45a5c810c741e6b41f8515be24c51121f8fc21f`, directory `examples/cfd/darcy_physics_informed`. The files `darcy_physics_informed_fno.py`, `utils.py`, `download_data.py` and `conf/config_pino.yaml` were fetched from `raw.githubusercontent.com` at that commit on 2026-10-24 and read line by line against this repository. The DeepONet script of the same directory is out of scope (it is not used here).

Status labels: **IDENTICAL** (same code path or values), **EQUIVALENT** (restructured, same computation), **DIFFERENT** (behaviour changes; stated in the README).

## Configuration

| Item | Upstream | This repository | Status | Evidence |
|---|---|---|---|---|
| `conf/config_pino.yaml` | `start_lr 0.001`, `gamma 0.99948708`, `max_epochs 50`, `physics_weight 0.1`, FNO block | upper part of `configs/official.yaml` | IDENTICAL | SHA-256 `40bbd9aa…d7ef` of the fetched file equals the checksum asserted in `tests/test_config.py` |
| Study configs | none | `local` (15 epochs, `gamma` 0.99829129), `full` (50 epochs, 3 seeds) | DIFFERENT (added) | `local` compresses the schedule: 0.99948708^5100 = 0.073054 and 0.99829129^1530 = 0.073054, the same final rate |

## Data

| Item | Upstream | This repository | Status |
|---|---|---|---|
| Source | Google Drive id `1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV`, `piececonst_r241_N1024_smooth1` | same | IDENTICAL |
| Train / validation | samples 0 to 101 / 102 to 203 (`split_percentage = [10, 10]`) | same | IDENTICAL |
| Test | none | samples 204 to 459 | DIFFERENT (added) |
| Crop | `[:, :240, :240]` | `[:, :, :240, :240]` | IDENTICAL |
| Scaling | `Kcoeff / 4.49996`, `sol / 3.88433e-3`, no shift | same | IDENTICAL |
| Model input | `invar[:, 0]` = `Kcoeff` (the `Kcoeff_x`, `Kcoeff_y` channels are loaded and unused) | `Kcoeff` only | EQUIVALENT |
| Coordinates | `x, y = linspace(0, 1, 240)` returned by the dataset, unused by the FNO | not built | EQUIVALENT for the FNO; note that these coordinates do not match the stored grid (cell centres `(i + 1/2) / 241`) |
| Loading | HDF5 read per sample | split loaded into memory once | EQUIVALENT |

## Model

| Item | Upstream | This repository | Status |
|---|---|---|---|
| Architecture | PhysicsNeMo `FNO(in 1, out 1, decoder 1 x 32, dim 2, latent 32, 4 layers, 12 modes, padding 9)` | `build_fno` with the same arguments; 2,365,169 parameters | IDENTICAL |
| Initialisation | unseeded | seeded, built on CPU, then moved | DIFFERENT (reproducibility) |

## PDE residual

| Item | Upstream | This repository | Status |
|---|---|---|---|
| Equation | `Diffusion(T="u", D="k", Q=forcing)` in `utils.py`, SymPy-expanded to `-k u_xx - k u_yy - k_x u_x - k_y u_y - Q` | class copied unchanged | IDENTICAL |
| Derivatives | `PhysicsInformer(grad_method="finite_difference")`: periodic second-order central differences of `u` and `k` on the grid (`physicsnemo.nn.functional.derivatives.uniform_grid_gradient`) | same module | IDENTICAL |
| Forcing | `1.0 * 4.49996 * 3.88433e-3 = 0.01748` | `shipped`: same; `consistent`: `1 / (4.49996 * 3.88433e-3) = 57.21` | `shipped` IDENTICAL, `consistent` DIFFERENT (see the audit, F1) |
| Spacing | `fd_dx = 1 / 240` ("unit square with resolution as 240") | `shipped`: same; `consistent`: `1 / 241` | `shipped` IDENTICAL, `consistent` DIFFERENT (F2) |
| Coefficient | `k = Kcoeff / 4.49996` (model input) | same in both definitions | IDENTICAL; the audit (F4) shows this is not the coefficient of the equation the data solve |
| Border | residual of the outer 2 rows and columns set to zero, still counted in the mean | same | IDENTICAL |
| Reduction | `F.l1_loss(residual, 0)` | same | IDENTICAL |
| Batch | `batch_size = 1`; the FD module uses `u[0, 0]` only | `batch_size = 1` enforced; larger batches raise | EQUIVALENT (guard added) |
| Device | CUDA or CPU | the residual runs on CPU when the FNO is on Apple MPS (Warp has no MPS backend) | DIFFERENT (device only; gradients flow through the transfer, `tests/test_physics.py::test_gradient_reaches_the_field`) |

## Loss, optimiser and training loop

| Item | Upstream | This repository | Status |
|---|---|---|---|
| Loss | `mse(outvar, out) + 1/240 * physics_weight * loss_pde` | same expression; for weight 0 the residual is not evaluated | IDENTICAL (`tests/test_pipeline.py::test_first_step_loss_is_the_upstream_expression`) |
| Optimiser | Adam, betas (0.9, 0.999), lr 1e-3, no weight decay | same | IDENTICAL |
| Schedule | `ExponentialLR(gamma)` stepped after every sample | same | IDENTICAL |
| Sample order | `DataLoader(shuffle=True)`, unseeded | `torch.randperm` with a seeded generator | EQUIVALENT (seeded) |
| Epochs | 50 | 50 in `official`/`full`, 15 in `local` | DIFFERENT for `local` |
| Checkpoint | every epoch | final epoch only; the final checkpoint is evaluated | EQUIVALENT for evaluation (no model selection in either) |

## Evaluation

| Item | Upstream | This repository | Status |
|---|---|---|---|
| Validation | MSE per epoch, figure of the last sample | MSE, relative L2, residual under both definitions per epoch | DIFFERENT (added) |
| Test | none | relative L2 per sample, MSE, residual statistics, timings | DIFFERENT (added) |
| Logging | `LaunchLogger` | CSV and JSON records | DIFFERENT (format only) |

## Observations about the upstream example

1. **Forcing.** The scaled equation requires `1 / (K_SCALE * U_SCALE) = 57.21`; the script uses the product `0.01748`. Under the shipped residual the reference pressure has a residual of 56.7 on every cell of constant permeability. Confirmed in this audit (`docs/SCIENTIFIC_AUDIT.md`, F1).
2. **Spacing.** The stored fields are on the cell centres `(i + 1/2) / 241`. The value 1/240 is the spacing of the node grid that the data-generation solver uses internally, not the spacing of the stored fields (F2). Effect on second differences: 0.8 %.
3. **Coefficient.** The data were generated from the two-valued field `coeff`; the residual uses the smoothed `Kcoeff` (F4).
4. **Batch.** `GradientsFiniteDifference.forward` differentiates `u[0, 0]`; with a batch size above one only the first sample would enter the residual. The example uses batch size 1 and is not affected.

None of these were reported upstream by this audit; doing so is a separate, outward-facing action left to the author.
