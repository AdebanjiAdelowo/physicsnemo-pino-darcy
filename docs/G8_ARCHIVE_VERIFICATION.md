# G8 frozen test results — independent archive verification

Verified 2026-11-09 from user-uploaded Kaggle output. No training or model evaluation was performed during this verification.

- Original evaluation commit: `82edeed582702a1c209f8b7a63d5cc9f69474ec3`.
- Documentation branch commit already CI-passed: `d720e719ee68830d6237f1a9338091ec5bdd31c8`.
- Files: `regularisation_study.json` plus eight original `*_variants_test.json` records. All nine SHA-256 hashes match `SHA256SUMS`.
- Primary, budget A (50 epochs), five paired seeds: PINO mean test relative L2 **0.030026093**, WD+VB **0.038191225**.
- Mean paired difference PINO − WD+VB: **-0.008165132**, relative change **-21.3796%**. All five differences negative.
- Study-reported paired t 95% CI: [-0.008833078, -0.007497186], p=4.4957106e-06; exact two-sided sign-test p=0.0625. Small n limits inferential certainty.
- Selected WD: 0.001, selected study `reg_cuda_wd1e-3`; this verification did not change selection.

## Scientific provenance limitations

Earlier preflight inspected test-set descriptive information before freeze. The test set must not be described as untouched. Some aggregate metadata reports CPU/null GPU, while individual raw test JSON records report CUDA Tesla T4; this discrepancy is documented, not silently resolved. Some raw records report dirty working trees. Neither this archive verification nor the CI run proves that every model-evaluation operation executed on GPU.

The original files are preserved byte-for-byte in this archive. The checksum manifest also lists other Kaggle outputs that are not included here; verify only the included nine data files against the manifest, rather than treating absent checkpoint archives as checksum failures.
