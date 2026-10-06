# Results

Tracked result files. Each study directory is a copy of a run directory without its checkpoints, written by `python scripts/results.py publish <run directory or archive>`.

| Path | Content |
|---|---|
| `data_check.json` | output of `scripts/verify_data.py`: content, grid, forcing and reference residual of the dataset |
| `<study>/study_metadata.json` | timestamp, git commit and dirty flag, upstream commit, device, package versions, residual definitions, resolved configuration, dataset checksums |
| `<study>/summary.csv` | one row per run |
| `<study>/summary.json` | the same rows, mean and standard deviation over seeds per setting, the residual of the reference solution, and the environment of the evaluation |
| `<study>/fields.npz` | permeability, reference pressure, predictions and residual fields of the plotted test sample |
| `<study>/<run>/history.csv` | per epoch: learning rate, data loss, PDE loss and total loss on the training set, validation error and validation residuals |
| `<study>/<run>/train_metrics.json` | residual definition, physics weight, seed, parameter count, timings, memory |
| `<study>/<run>/eval_metrics.json` | validation and test errors, residual statistics under both definitions, per-sample test errors, inference time |

Run names: `dataonly_seed<S>` for weight 0, otherwise `<residual>_w<weight>_seed<S>` with `<residual>` the definition used in the training loss (`shipped` or `consistent`).

| Study | Configuration | Status |
|---|---|---|
| `data_check.json` | 460 samples of the dataset, CPU | VERIFIED |
| `local` | `configs/local.yaml`: 15 epochs, 1 seed, Apple MPS | LOCAL/REDUCED |
| `full` | `configs/full.yaml`: 50 epochs, 3 seeds, CUDA | NOT RUN |

Units: all losses and residuals are in the normalised variables $k' = k / 4.49996$ and $u' = u / 3.88433 \times 10^{-3}$. `*_pde_l1_*` is the mean absolute residual over all 240 × 240 cells with the two outer rows and columns set to zero, which is the quantity in the training loss; the other residual statistics are taken over the interior cells. `test_rel_l2_*` are relative $L^2$ errors as fractions. Times are wall-clock with the device synchronised. Memory fields are CUDA peak allocations and are empty on other devices.

Notes on `local`: the machine was running other jobs during the study, so its training and inference times vary between runs for reasons unrelated to the settings. `study_metadata.json` lists `evaluation.plot_samples: 3`, the value at training time; the tracked `fields.npz` was written by a later evaluation of the same checkpoints with one plotted sample.
