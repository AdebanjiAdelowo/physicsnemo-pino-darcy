# Reproducibility and verification checklist

State on 2026-10-24, audit of the repository at the commit that adds this file. `[x]` checked in the audit, with the evidence named; `[ ]` not done.

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
- [ ] Re-training a run and comparing histories (bitwise or to tolerance) on MPS.
- [ ] More than one training seed (`docs/EXPERIMENT_GAPS.md`, G1).
- [ ] The 50-epoch study (`configs/full.yaml`, G2).

## Provenance of the audit files

- [x] Historical artifacts (`results/local/`, `results/data_check.json`, `figures/`) unchanged by the audit.
- [x] New results are in `results/audit/` and were written from a committed, clean tree (see their `environment.git_commit` and `git_dirty`).
