"""Split files and normalisation, on a small synthetic copy of the dataset layout."""

import json
import zipfile

import h5py
import numpy as np
import pytest
import scipy.io

from pino_darcy import data as D
from pino_darcy.config import load_config


@pytest.fixture
def fake_root(tmp_path):
    rng = np.random.default_rng(0)
    cfg = load_config("official").data
    mat = {key: rng.random((1024, 5, 5)) for key in D.FIELDS}
    mat["coeff"] = rng.random((1024, 5, 5)).astype(np.float32)
    (tmp_path / "datasets").mkdir()
    scipy.io.savemat(tmp_path / D.MAT_NAME, mat)
    with zipfile.ZipFile(tmp_path / "datasets" / "Darcy_241.zip", "w") as z:
        z.write(tmp_path / D.MAT_NAME, D.MAT_NAME)
    return tmp_path, cfg, mat


def test_split_bounds_follow_upstream():
    cfg = load_config("official").data
    assert D.split_bounds(cfg) == {"train": (0, 102), "validation": (102, 204), "test": (204, 460)}
    cfg.test_samples = 900
    with pytest.raises(ValueError):
        D.split_bounds(cfg)


def test_prepare_writes_consecutive_float32_splits(fake_root):
    root, cfg, mat = fake_root
    manifest = D.prepare(cfg, root)
    data_dir = root / cfg.dir
    with h5py.File(data_dir / "validation.hdf5") as f:
        assert set(f.keys()) == {*D.FIELDS, "coeff"}
        assert f["sol"].shape == (102, 1, 5, 5) and f["sol"].dtype == np.float32
        assert np.array_equal(f["sol"][:, 0], mat["sol"][102:204].astype(np.float32))
    with h5py.File(data_dir / "test.hdf5") as f:
        assert np.array_equal(f["Kcoeff"][0, 0], mat["Kcoeff"][204].astype(np.float32))
    assert manifest["files"]["train.hdf5"] == D.sha256(data_dir / "train.hdf5")
    assert not (data_dir / D.MAT_NAME).exists()
    assert json.loads((data_dir / "manifest.json").read_text()) == manifest

    stamp = (data_dir / "train.hdf5").stat().st_mtime_ns
    assert D.prepare(cfg, root) == manifest  # second call: nothing is rewritten
    assert (data_dir / "train.hdf5").stat().st_mtime_ns == stamp


def test_load_split_normalises_as_upstream(fake_root):
    root, cfg, mat = fake_root
    D.prepare(cfg, root)
    split = D.load_split(root / cfg.dir / "train.hdf5", 7)
    assert split["k"].shape == (7, 1, 5, 5) and split["u"].dtype.is_floating_point
    assert np.allclose(split["k"][:, 0].numpy(), mat["Kcoeff"][:7].astype(np.float32) / 4.49996e00)
    assert np.allclose(split["u"][:, 0].numpy(), mat["sol"][:7].astype(np.float32) / 3.88433e-03)
    with pytest.raises(ValueError):
        D.load_split(root / cfg.dir / "train.hdf5", 500)
