"""Integrity checks for the frozen G8 test-results archive (no model evaluation)."""

from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "g8_frozen_test"
DATA_FILES = {
    "regularisation_study.json",
    "full_cuda_variants_test.json",
    "full_cuda_more_variants_test.json",
    "reg_cuda_wd0_variants_test.json",
    "reg_cuda_wd1e-3_variants_test.json",
    "reg_cuda_wd1e-2_variants_test.json",
    "reg_cuda_wd1e-1_variants_test.json",
    "reg_cuda_wd1e0_variants_test.json",
    "reg_cuda_wd1e1_variants_test.json",
}


def test_frozen_g8_json_bytes_match_kaggle_manifest():
    """Check only the nine archived JSONs; the original manifest lists unarchived bundles."""
    lines = (ARCHIVE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries = {}
    for line in lines:
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        assert name not in entries, f"duplicate manifest entry: {name}"
        entries[name] = digest
    assert DATA_FILES <= entries.keys()
    for name in sorted(DATA_FILES):
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == entries[name], name


def test_frozen_g8_archive_has_only_expected_files():
    expected = DATA_FILES | {"SHA256SUMS", "stage_g8.sh"}
    actual = {p.name for p in ARCHIVE.iterdir() if p.is_file()}
    assert actual == expected
    assert all(p.is_file() for p in ARCHIVE.iterdir())
