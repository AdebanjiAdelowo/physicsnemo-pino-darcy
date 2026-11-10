# v1.0.0 — research release notes (draft)

**Status:** Proposed release; not published or tagged. Review and confirm the release target after post-merge CI passes.

**Release target:** `3dfcc57bc542690eca52402b7a80ffbcececa8e7` (the merged research and documentation state). This notes file is prepared separately and is not part of that frozen target commit.

## Scope

A research implementation and audit of physics-informed Fourier Neural Operators for steady Darcy flow, based on NVIDIA PhysicsNeMo's Darcy example. The project studies physics-residual weighting, the dataset/operator mismatch, discretisation choices, label efficiency, five-seed MPS/CUDA replication, and follow-up conventional-regularisation controls (G8).

## Selected results

- The upstream residual forcing is inconsistent with the normalized dataset: the source uses approximately 0.01748 whereas the consistent normalized forcing is approximately 57.21. The corrected cell-grid residual remains an imperfect proxy near permeability jumps.
- At 5100 optimizer steps and five paired seeds, the corrected residual at weight 0.1 reduced mean test relative L2 error against data-only training by approximately 21.6% on Apple MPS and 21.5% on an NVIDIA Tesla T4 (CUDA). The CUDA replication covers the primary comparison, not the full weight/discretisation/label sweeps.
- In G8's follow-up five-seed comparison, PINO mean test relative L2 error was 0.0300261 versus 0.0381912 for the selected weight-decay-plus-validation-best control (WD 0.001), a 21.38% relative difference. The reported paired Student-t 95% interval for the absolute difference was [-0.0088331, -0.0074972]; exact two-sided sign-test p = 0.0625. This is not independent holdout confirmation.

## Reproducibility and provenance

- Source, configurations, offline tests, analysis scripts, tracked result records, scientific audit and reproduction instructions are in this repository; see [README](../README.md), [results](../results/README.md), [audit](../docs/SCIENTIFIC_AUDIT.md), and [reproducibility checklist](../docs/REPRODUCIBILITY_CHECKLIST.md).
- The dataset is downloaded separately and is not redistributed. Full checkpoints and some external Kaggle ZIP archives are not included in Git; availability of those external artifacts must not be assumed.
- The G8 archive verification reconciled nine frozen JSON records against SHA-256 entries. It did not independently rerun GPU training, regenerate figures or verify unavailable checkpoint archives.
- Earlier G8 preflight accessed test-split descriptive information before the selection freeze. The historical `test_set_read: false` field must not be interpreted as proof of an untouched holdout. The same test split had been used in earlier studies. Some records carry dirty-tree flags and inconsistent aggregate device metadata; see audit F19 and [G8 archive verification](../docs/G8_ARCHIVE_VERIFICATION.md).
- Findings are specific to one dataset, FNO architecture, splits and hardware configurations; no cross-dataset generalisation is claimed.

## Release gate

- [ ] Post-merge CI on target commit `3dfcc57bc542690eca52402b7a80ffbcececa8e7` completed successfully.
- [ ] Confirm release notes, source tree, license and link rendering.
- [ ] Create annotated `v1.0.0` tag **on the target commit**, not on the documentation-preparation branch.
- [ ] Publish GitHub release with these notes, without claiming unavailable checkpoint artifacts are attached.

This file is a proposed release-note draft. Checking boxes or publishing a release requires a separate verified action.
