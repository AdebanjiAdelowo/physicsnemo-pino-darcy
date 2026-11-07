# Reproducibility and verification checklist

State on 2026-10-24, audit of the repository at the commit that adds this file. `[x]` checked in the audit, with the evidence named; `[ ]` not done. Items checked later carry their date; the section "Update 2026-11-07" covers the 50-epoch studies (G2 to G6).

## Code and environment

- [x] Upstream commit pinned in `requirements.txt` and recorded in every result file (`physicsnemo.commit`).
- [x] Upstream configuration unchanged: SHA-256 of `conf/config_pino.yaml` fetched at the pinned commit equals the checksum in `tests/test_config.py`.
- [x] Exact package versions of the study environment: `requirements.lock` (`uv pip freeze`).
- [ ] Lockfile resolved from `pyproject.toml` (`uv lock`): the project declares its dependencies in `requirements.txt` only.
- [x] Results record the git commit and a dirty flag: training of `results/local` at `d1bd2ca` (clean), evaluation at `9ebcaab` (clean); no change to training, data or metric code between them (`git diff d1bd2ca 9ebcaab -- src`).

## Tests and static checks

- [x] `python -m pytest -q`: 48 passed (41 existing, 7 added in `tests/test_reference.py`), offline, CPU, under 30 s.
- [x] `ruff check src scripts tests --select E,F,W`: no findings. With `B`: nine `B905` (`zip` without `strict=`), stylistic.
- [x] Tests are deterministic and need neither the dataset nor a GPU (synthetic fields, fixed seeds).

## Data

- [x] Archive and split checksums recorded (`results/data_check.json`) and identical on re-preparation.
- [x] Splits disjoint by index (0 to 101, 102 to 203, 204 to 459) and by content (no identical `coeff` field across the 460 samples).
- [x] Reference pressure regenerated from `coeff` with a re-implementation of the original solver (`scripts/verify_reference.py`, `results/audit/reference_check.json`).
- [ ] Origin of the normalisation constants 4.49996 and 3.88433e-3 (they match the standard deviations of `coeff` and `sol`; over which samples they were computed is not documented upstream).

## Results

- [x] `summary.csv` and `summary.json` recomputed from `eval_metrics.json`, `train_metrics.json` and `history.csv` of every run: agreement to 1e-6 relative, except the median, which is the lower median by construction (`metrics.summarise`).
- [x] Every number in the README results tables traced to the result files.
- [x] Checkpoints of the reduced study re-evaluated: test relative L2 reproduced (`results/audit/local_solver_residual.json`, `stored_rel_l2_mean` against `rel_l2_mean`).
- [x] Re-training a run and comparing histories (bitwise or to tolerance) on MPS. (2026-11-06: one-epoch repeats of data only and `consistent` 0.1 give identical parameters on this machine, `results/audit/determinism.json`; full-length runs and other residuals not repeated.)
- [x] More than one training seed at weight 0.1: seeds 0, 1, 2 (`docs/EXPERIMENT_GAPS.md`, G1; `docs/SCIENTIFIC_AUDIT.md`, F14). Other weights: one seed at 15 epochs; three seeds at every weight at 50 epochs (2026-11-07, `full_mps_sweep`).
- [x] The 50-epoch study (G2). (2026-11-07: on MPS with `configs/full_mps.yaml` and `full_mps_sweep.yaml`, audit F15; the CUDA configuration `configs/full.yaml` has not been run.)

## Provenance of the audit files

- [x] Historical artifacts (`results/local/`, `results/data_check.json`, `figures/`) unchanged by the audit.
- [x] New results are in `results/audit/` and were written from a committed, clean tree (see their `environment.git_commit` and `git_dirty`).

## Update 2026-11-07: the 50-epoch studies (G2 to G6)

- [x] Analysis plans committed before any of their runs: `docs/FULL_BUDGET_PLAN.md` (`eaca64c`), `docs/OPERATOR_AND_LABELS_PLAN.md` (`beaac7f`).
- [x] Every study trained from a git worktree at its plan's commit. `full_mps` and `full_mps_operator` record `git_dirty: false`; the other studies record `git_dirty: true` caused only by untracked audit outputs of the same queue in the worktree (`git diff HEAD` empty there).
- [x] Every run complete: configuration, seed, 5100 optimiser steps, full history, checkpoint (not tracked), test metrics. One run (`full_mps_sweep/consistent_w0.001_seed2`) was interrupted by the end of a session before writing any record and was retrained from scratch.
- [x] Stored test errors equal the errors recomputed from the checkpoints by `scripts/evaluate_solver_residual.py`, to 1e-5 relative, for every run (asserted in `scripts/full_study.py`).
- [x] Torch `solver` residual equals the NumPy re-implementation of the generator's solver in float64 on real fields; gradient checked against central differences (`results/audit/solver_operator_check.json`).
- [x] G5 unlabelled pool: training-split samples only, permeability only (`src/pino_darcy/engine.py`, `tests/test_config.py`, `tests/test_operator_training.py`).
- [x] `python -m pytest -q`: 88 passed, offline, CPU. `ruff check src scripts tests`: no findings.
- [x] Historical records (`results/local/`, `results/local_seeds/`, earlier figures and audit files) unchanged (`git diff` against the earlier commits is empty for them).
- [ ] Training on CUDA (the hardware the G2 plan targeted); all 50-epoch runs used Apple MPS.
- [ ] Bitwise repeat of full-length runs, or of the `solver` and unlabelled-pool code paths.
