# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from examples/cfd/darcy_physics_informed/utils.py and download_data.py
# of NVIDIA PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f.
# Modified by Adebanji Adelowo (2026): the Google Drive id, the file name, the
# normalisation constants, the 240 x 240 crop and the first two splits (samples
# 0-101 for training, 102-203 for validation) are those of the original. The
# .mat file is converted straight into the split files instead of through a
# full HDF5 copy, a third split (test) is added, checksums are recorded, and a
# split is loaded into memory once instead of being read from disk per sample.

import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch

# Upstream: _FNO_datatsets_ids["Darcy_241"] and _FNO_dataset_names["Darcy_241"][0] in utils.py.
DRIVE_ID = "1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV"
MAT_NAME = "piececonst_r241_N1024_smooth1.mat"
# Upstream: the constants of HDF5MapStyleDataset.__getitem__ in utils.py.
K_SCALE = 4.49996e00
U_SCALE = 3.88433e-03
CROP = 240
FIELDS = ("Kcoeff", "Kcoeff_x", "Kcoeff_y", "sol")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(data_dir: Path) -> Path:
    """Download ``Darcy_241.zip`` from the FNO authors' Google Drive folder, as upstream does."""
    import gdown

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir.parent / "Darcy_241.zip"  # upstream: f"{outdir}{name}.zip"
    if not archive.exists():
        gdown.download(id=DRIVE_ID, output=str(archive))
    if not archive.exists() or not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive} was not downloaded. Fetch Darcy_241.zip by hand from the Google Drive folder "
            "named in the README and place it there."
        )
    return archive


def split_bounds(cfg_data) -> dict:
    """Index ranges of the splits in the 1024 samples of the file.

    ``train`` and ``validation`` are upstream's ``split_percentage = [10, 10]``;
    ``test`` follows them.
    """
    n = 1024
    n_train = int(n * cfg_data.train_percent / 100)
    n_val = int(n * cfg_data.validation_percent / 100)
    stop = n_train + n_val + cfg_data.test_samples
    if stop > n:
        raise ValueError(f"the splits need {stop} samples; the file has {n}.")
    return {"train": (0, n_train), "validation": (n_train, n_train + n_val), "test": (n_train + n_val, stop)}


def prepare(cfg_data, root: Path) -> dict:
    """Write ``train.hdf5``, ``validation.hdf5`` and ``test.hdf5`` and a manifest. Idempotent."""
    data_dir = Path(root) / cfg_data.dir
    bounds = split_bounds(cfg_data)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["splits"] == {k: list(v) for k, v in bounds.items()} and all(
            (data_dir / f"{split}.hdf5").exists() for split in bounds
        ):
            return manifest
    archive = download(data_dir)
    with zipfile.ZipFile(archive) as z:
        z.extract(MAT_NAME, data_dir)
    mat = scipy.io.loadmat(data_dir / MAT_NAME)
    manifest = {
        "source": f"https://drive.google.com/uc?id={DRIVE_ID}",
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "mat_file": MAT_NAME,
        "mat_keys": {k: list(v.shape) for k, v in mat.items() if not k.startswith("__")},
        "splits": {k: list(v) for k, v in bounds.items()},
        "files": {},
    }
    for split, (start, stop) in bounds.items():
        path = data_dir / f"{split}.hdf5"
        with h5py.File(path, "w") as f:
            for key in manifest["mat_keys"]:
                # N, C, H, W in float32, as upstream's preprocess_FNO_mat
                f.create_dataset(key, data=np.expand_dims(mat[key][start:stop], axis=1), dtype="float32")
        manifest["files"][path.name] = sha256(path)
    (data_dir / MAT_NAME).unlink()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(path: Path, n_samples: int | None = None) -> dict:
    """Normalised tensors of one split, on CPU.

    Returns
    -------
    dict
        ``k`` and ``u`` of shape ``[N, 1, 240, 240]``: permeability divided by ``K_SCALE``
        and pressure divided by ``U_SCALE``, cropped from the 241 x 241 grid as upstream.
    """
    with h5py.File(path, "r") as f:
        k = np.asarray(f["Kcoeff"][:n_samples, :, :CROP, :CROP])
        u = np.asarray(f["sol"][:n_samples, :, :CROP, :CROP])
    if n_samples is not None and len(k) < n_samples:
        raise ValueError(f"{path} holds {len(k)} samples; {n_samples} were requested.")
    return {"k": torch.from_numpy(k) / K_SCALE, "u": torch.from_numpy(u) / U_SCALE}
