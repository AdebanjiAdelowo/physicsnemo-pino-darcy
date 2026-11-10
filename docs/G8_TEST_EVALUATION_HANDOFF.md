# G8 frozen test evaluation — audit handoff (9 November 2026)

## Scope and source integrity

This record documents the one authorized Kaggle `MODE=test` session for the regularisation-controls study at commit `82edeed582702a1c209f8b7a63d5cc9f69474ec3`. It is an audit handoff, **not** a replacement for the original frozen selection, the raw per-variant test records, or the original machine-generated `regularisation_study.json`. The latter must be archived byte-for-byte.

The Kaggle execution log records a successful end-of-session SHA-256 verification of the evaluation outputs and eight checkpoint bundles, ending with `KAGGLE REGULARISATION SESSION COMPLETE.` The preflight records `freeze_authenticated=true`, `dataset_hashes_verified=true`, `regularisation_archives_verified=12`, `historical_archives_verified=4`, and successful pytest/Ruff. The test session log records `185 passed, 1 skipped` and `All checks passed!`. The working tree of the research repository was not modified in order to produce this handoff.

Locally received source artifacts and independently calculated SHA-256:
- `regularisation_study.json` (received as `regularisation_study.txt`): `604c3e71b59b11f92d41736f858cd19a4b98b2615468f1ae061c13455f76cacd`
- `preflight.json`: `881eeb90eab10baafef6d44d87a36ee38c0de60b2651d573c3f2527bcdaeb4b0`
- `environment.txt`: `dc9064d433e996226947e74b0f4f3caef762007f10f47235c58e338827fc8e03`
- `gpu.txt`: `f243c899e3510419ed446fbff4d8b2749ad8ec2d5faa2c177476cc9f5e427e09`
- Received notebook text log `download.txt`: `1e70a9a599fd9d7954adb16d9d5d515a1218fe79d0721d9fbef05ab465c9f38a`

The hashes above authenticate the **received bytes**, not the unreceived raw test records or checkpoint ZIPs. The Kaggle log reports their checks as successful; independent local verification of those additional files remains outstanding.

## Primary result

Frozen primary comparison: corrected PINO minus AdamW weight decay `0.001` plus validation-best checkpoint (`WD+VB`), budget A, five paired seeds, test relative L2. Lower is better.

- Mean WD+VB: `0.03819122537970543`.
- Mean corrected PINO: approximately `0.030026093497872354` (derived by adding the reported mean paired difference).
- Mean paired difference: `-0.008165131881833076`.
- Paired Student-t 95% CI: `[-0.008833077939116427, -0.0074971858245497245]`; two-sided p `4.495710633078457e-06`.
- Relative improvement against mean WD+VB: `21.37960172959513%`, CI `[19.630650103554104%, 23.128553355636156%]`.
- All five differences are negative. The exact two-sided sign-test p-value is `0.0625`; avoid presenting the five seeds as a large independent sample.
- The confidence interval excludes zero and is well beyond the registered ±5% practical-relevance threshold. In the tested setting, WD+VB did not match corrected PINO.

This is a **follow-up analysis**, not a fresh untouched-holdout confirmation; the test set was examined in prior work.

## Runtime and provenance caveats

The notebook log establishes a Tesla T4 CUDA-capable environment: a runtime assertion reports one visible `Tesla T4`, and the pinned venv reports PyTorch `2.15.0+cu132` and CUDA `13.2` matching historical `full_cuda`. The `nvidia-smi` inventory shows two T4 GPUs. This does **not** independently establish the execution device of every evaluation kernel.

The study JSON's `environment` object reports `device: cpu`, `gpu_name: null`, `cuda_version: null`, and `git_dirty: true`. These are retained verbatim. A plausible explanation is that the study aggregation was a CPU-only process after CUDA evaluation, and generated files made the tree dirty, but neither explanation is proven by the received artifacts. Do not silently replace these values or claim the discrepancies have been resolved.

The original frozen selection contains `test_set_read: false`, although earlier `train_select` code eagerly loaded test tensors and preflight computed descriptive test diagnostics before selection. The access deviation is documented in `docs/SCIENTIFIC_AUDIT.md` F19. There is no evidence in the reviewed records of new test-model performance evaluation before selection, but unrecorded human exposure cannot be ruled out. Do not claim the test holdout remained unread.

## Release boundary

No retraining, reselection, or repeated test evaluation is authorized by this handoff. Before a final results commit, acquire the **original** eight `*_variants_test.json` files and `regularisation_study.json` from the frozen Kaggle output, verify each against `SHA256SUMS`, and inspect the exact staged Git diff. Preserve immutable freeze records. Commit only verified outputs and documentation to `research/regularization-controls`; run CI and review the resulting commit before any merge. This handoff alone is not the complete results archive.

## Post-handoff archival completion (9 November 2026)

The paragraphs above are the original handoff and describe what was outstanding **at the time of handoff**. Subsequently, the original eight `*_variants_test.json` records and `regularisation_study.json` were archived under `results/g8_frozen_test/` at commit `fcf67e3ec48df99b2af3f8197f1579982160dccb`. All nine archived JSON SHA-256 digests matched the supplied `SHA256SUMS` during staging, as recorded in `docs/G8_ARCHIVE_VERIFICATION.md`. The original manifest also lists checkpoint bundles and other evidence not included in this nine-file verification; their independent local verification is **not** implied by the JSON check.

Commit `b3cb401bcfc50eb158b9297bebd0559c68365d41` added `tests/test_g8_archive_integrity.py` to enforce the nine JSON hashes and the expected archive file inventory in CI. The push and pull-request workflows for that commit both passed. These checks establish integrity against the archived manifest, not independent retraining, independent GPU-kernel execution verification, or an untouched test holdout. The original freeze record, result JSONs, and statistical conclusions were not changed.

This addendum supersedes only the handoff's **pending archive** instructions; it does not alter the historical provenance or limitations stated above. The pull request remains subject to review of the broader branch before merge.
