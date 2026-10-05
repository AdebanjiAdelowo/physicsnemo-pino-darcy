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
# The training loop of ``train_run`` (FNO, MSE data loss, L1 PDE loss weighted by
# 1 / 240 * physics_weight, Adam, ExponentialLR stepped after every sample,
# validation after every epoch, checkpoint) is adapted from
# examples/cfd/darcy_physics_informed/darcy_physics_informed_fno.py of NVIDIA
# PhysicsNeMo, commit b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by
# Adebanji Adelowo (2026): seeding, device selection with Apple MPS, in-memory
# data, the residual definition as an argument, no residual evaluation when the
# weight is zero, relative L2 and residual metrics on fixed validation and test
# sets, synchronised timing, CSV/JSON records, and the loop over residual
# definitions, weights and seeds. The LaunchLogger console logging and the
# per-epoch validation figure of the original are not used.

import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint

from . import data as D
from . import physics as P
from .metrics import relative_l2, summarise
from .model import build_fno, count_parameters
from .provenance import collect_metadata, write_json

HISTORY_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss_data",
    "train_loss_pde",
    "train_loss_total",
    "validation_mse",
    "validation_rel_l2",
    "validation_pde_l1_shipped",
    "validation_pde_l1_consistent",
    "elapsed_seconds",
]


def resolve_device(name: str) -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. An explicit name must be available."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but CUDA is not available on this machine.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps was requested but MPS is not available on this machine.")
    return device


def synchronise(device: torch.device) -> None:
    """Wait for queued accelerator work so that a wall-clock reading is meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_name(residual: str | None, weight: float, seed: int) -> str:
    """``dataonly_seed0`` for a zero weight, otherwise e.g. ``shipped_w0.1_seed0``."""
    if weight == 0:
        return f"dataonly_seed{seed}"
    return f"{residual}_w{weight:g}_seed{seed}"


def study_runs(cfg: DictConfig) -> list[tuple[str | None, float, int]]:
    """(residual, weight, seed) of every run. The data-only model does not depend on the residual."""
    runs = []
    weights = [float(w) for w in cfg.study.physics_weights]
    for seed in cfg.study.seeds:
        if 0.0 in weights:
            runs.append((None, 0.0, int(seed)))
        for residual in cfg.study.residuals:
            runs += [(str(residual), w, int(seed)) for w in weights if w != 0]
    return runs


def validate_study(cfg: DictConfig) -> None:
    for residual in cfg.study.residuals:
        if residual not in P.RESIDUALS:
            raise ValueError(f"unknown residual {residual!r}; choose from {sorted(P.RESIDUALS)}.")
    if any(float(w) < 0 for w in cfg.study.physics_weights):
        raise ValueError("physics weights must be non-negative.")
    if cfg.batch_size != 1:
        raise ValueError("batch_size must be 1: the finite-difference residual of PhysicsInformer uses one sample.")
    names = [run_name(*run) for run in study_runs(cfg)]
    if len(set(names)) != len(names):
        raise ValueError("two runs of the study share a name.")


def load_datasets(cfg: DictConfig, root: Path, with_train: bool = True) -> dict:
    """Normalised CPU tensors of the splits, after preparing the files if necessary."""
    manifest = D.prepare(cfg.data, root)
    data_dir = Path(root) / cfg.data.dir
    sets = {split: D.load_split(data_dir / f"{split}.hdf5") for split in ("validation", "test")}
    if with_train:
        sets["train"] = D.load_split(data_dir / "train.hdf5", cfg.data.train_samples)
    sets["manifest"] = manifest
    return sets


@torch.no_grad()
def predict(model, k: torch.Tensor, device: torch.device, batch_size: int = 16) -> torch.Tensor:
    """Model output for a CPU tensor of normalised permeabilities, returned on CPU."""
    model.eval()
    return torch.cat([model(k[i : i + batch_size].to(device)).cpu() for i in range(0, len(k), batch_size)])


@torch.no_grad()
def residual_fields(informer, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Residual of every sample of CPU tensors ``u`` and ``k``, on CPU."""
    where = informer.device
    return torch.cat([P.residual_field(informer, u[i : i + 1].to(where), k[i : i + 1].to(where)).cpu() for i in range(len(u))])


def make_informers(device: torch.device) -> dict:
    return {name: P.build_informer(name, device) for name in P.RESIDUALS}


def train_run(
    cfg: DictConfig, residual: str | None, weight: float, seed: int, run_dir: Path, device: torch.device, sets: dict
) -> dict:
    """Train one FNO and write ``history.csv``, ``train_metrics.json`` and a checkpoint."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # The model is initialised on CPU from the seed and then moved, so the initial
    # weights do not depend on the device.
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_fno(cfg).to(device)
    informers = make_informers(device)
    phy_informer = informers[residual] if weight > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=cfg.start_lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    train_k, train_u = sets["train"]["k"], sets["train"]["u"]
    val_k, val_u = sets["validation"]["k"], sets["validation"]["u"]
    shuffle = torch.Generator().manual_seed(seed)

    history = []
    optim_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronise(device)
    start = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        sums = {"data": 0.0, "pde": 0.0, "total": 0.0}
        synchronise(device)
        tick = time.perf_counter()
        for i in torch.randperm(len(train_k), generator=shuffle).tolist():
            optimizer.zero_grad()
            invar = train_k[i : i + 1].to(device)
            outvar = train_u[i : i + 1].to(device)

            # Compute forward pass
            out = model(invar)

            # Compute data loss
            loss_data = F.mse_loss(outvar, out)

            # Compute total loss
            if phy_informer is None:
                loss_pde = None
                loss = loss_data
            else:
                loss_pde = P.pde_loss(P.residual_field(phy_informer, out, invar))
                loss = loss_data + P.PDE_LOSS_FACTOR * weight * loss_pde

            # Backward pass and optimizer and learning rate update
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums["data"] += loss_data.detach()
            sums["total"] += loss.detach()
            if loss_pde is not None:
                sums["pde"] += loss_pde.detach()
        synchronise(device)
        optim_seconds += time.perf_counter() - tick

        sums = {key: float(value) / len(train_k) for key, value in sums.items()}
        pred = predict(model, val_k, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss_data": sums["data"],
            "train_loss_pde": sums["pde"] if phy_informer is not None else None,
            "train_loss_total": sums["total"],
            "validation_mse": F.mse_loss(val_u, pred).item(),
            "validation_rel_l2": relative_l2(pred, val_u).mean().item(),
        }
        for name, informer in informers.items():
            row[f"validation_pde_l1_{name}"] = residual_fields(informer, pred, val_k).abs().mean().item()
        synchronise(device)
        row["elapsed_seconds"] = time.perf_counter() - start
        history.append(row)
        if not np.isfinite(row["train_loss_total"]):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}")

    synchronise(device)
    train_seconds = time.perf_counter() - start
    save_checkpoint(str(run_dir / "checkpoints"), models=model, optimizer=optimizer, scheduler=scheduler, epoch=cfg.max_epochs)

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    steps = cfg.max_epochs * len(train_k)
    metrics = {
        "run": run_dir.name,
        "residual": residual,
        "physics_weight": float(weight),
        "effective_pde_coefficient": float(weight) * P.PDE_LOSS_FACTOR,
        "seed": int(seed),
        "parameters": count_parameters(model),
        "optimiser_steps": steps,
        "final_train_loss_data": history[-1]["train_loss_data"],
        "final_train_loss_pde": history[-1]["train_loss_pde"],
        "final_train_loss_total": history[-1]["train_loss_total"],
        "train_seconds": train_seconds,
        "optimisation_seconds": optim_seconds,
        "optimisation_ms_per_step": 1e3 * optim_seconds / steps,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
    }
    write_json(run_dir / "train_metrics.json", metrics)
    return metrics


def evaluate_run(cfg: DictConfig, run_dir: Path, device: torch.device, sets: dict, write: bool = True) -> tuple[dict, dict]:
    """Evaluate the checkpoint of ``run_dir`` on the validation and test sets.

    Returns the metrics and, for the plotting samples of the test set, the prediction and the
    residual fields. ``write=False`` leaves ``eval_metrics.json`` untouched.
    """
    run_dir = Path(run_dir)
    model = build_fno(cfg).to(device)
    epoch = load_checkpoint(str(run_dir / "checkpoints"), models=model, device=device)
    if epoch != cfg.max_epochs:
        raise RuntimeError(f"{run_dir}: checkpoint is at epoch {epoch}, expected {cfg.max_epochs}.")
    model.eval()
    informers = make_informers(device)

    out = {"checkpoint_epoch": int(epoch)}
    for split in ("validation", "test"):
        k, u = sets[split]["k"], sets[split]["u"]
        pred = predict(model, k, device)
        rel = relative_l2(pred, u)
        out[f"{split}_mse"] = F.mse_loss(u, pred).item()
        out[f"{split}_rel_l2"] = summarise(rel)
        residuals = {name: residual_fields(informer, pred, k) for name, informer in informers.items()}
        for name, field in residuals.items():
            out[f"{split}_residual_{name}"] = P.residual_statistics(field, k)
        if split == "test":
            out["test_rel_l2_per_sample"] = rel.tolist()
            n = cfg.evaluation.plot_samples
            fields = {"prediction": pred[:n, 0].numpy(), **{f"residual_{name}": r[:n, 0].numpy() for name, r in residuals.items()}}

    # Inference time of one sample, forward pass only, synchronised, after warm-up.
    sample = k[:1].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.no_grad():
        for i in range(cfg.evaluation.timing_warmup + cfg.evaluation.timing_repeats):
            synchronise(device)
            tick = time.perf_counter()
            model(sample)
            synchronise(device)
            if i >= cfg.evaluation.timing_warmup:
                times.append(time.perf_counter() - tick)
    out["inference_ms_per_sample"] = 1e3 * statistics.median(times)
    out["peak_inference_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    if write:
        write_json(run_dir / "eval_metrics.json", out)
    return out, fields


def reference_residuals(sets: dict, device: torch.device) -> dict:
    """Residual statistics of the reference pressure of the dataset, per split and residual definition."""
    informers = make_informers(device)
    return {
        split: {
            name: P.residual_statistics(residual_fields(informer, sets[split]["u"], sets[split]["k"]), sets[split]["k"])
            for name, informer in informers.items()
        }
        for split in ("validation", "test")
    }


def prepare_study(cfg: DictConfig, root: Path, write_metadata: bool, with_train: bool = True) -> tuple[Path, torch.device, dict]:
    """Validate the config, resolve the device, load the data, optionally write ``study_metadata.json``."""
    validate_study(cfg)
    device = resolve_device(cfg.device)
    out_dir = Path(root) / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_metadata(device)  # before the data is prepared: the code state at start
    sets = load_datasets(cfg, root, with_train=with_train)
    if write_metadata:
        metadata["physics_device"] = str(P.physics_device(device))
        metadata["residual_definitions"] = P.RESIDUALS
        metadata["pde_loss_factor"] = P.PDE_LOSS_FACTOR
        metadata["config"] = OmegaConf.to_container(cfg, resolve=True)
        metadata["dataset"] = sets["manifest"]
        metadata["samples"] = {split: len(sets[split]["k"]) for split in ("train", "validation", "test") if split in sets}
        write_json(out_dir / "study_metadata.json", metadata)
    return out_dir, device, sets
