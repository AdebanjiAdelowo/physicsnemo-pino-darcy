"""Optimiser choice and validation-based early stopping for the regularisation controls.

Early stopping here never looks at the test set. The rule is applied to the validation relative
L2 error recorded after every epoch:

- an epoch *improves* if its validation error is below ``best * (1 - min_delta)``, where ``best``
  is the lowest validation error so far (``min_delta`` is relative; 0 means any decrease);
- training *stops* at the first epoch ``e`` with ``e - best_epoch >= patience`` (a patience of None
  never stops: the rule then selects the best validation epoch within the cap);
- the *selected* model is the one of ``best_epoch`` at the stop (or at the cap, whichever comes
  first).

The rule depends only on the validation errors up to the current epoch, so a run that is trained
past its stopping epoch passes through exactly the state that a run halted there would have had.
``engine.train_run`` therefore trains to ``max_epochs`` once and records, for every cap, the model
and the compute that early stopping with that cap would have produced.
"""

import math

import torch
from omegaconf import DictConfig

OPTIMISERS = ("adam", "adamw")


def optimiser_settings(cfg: DictConfig) -> tuple[str, float]:
    """``(name, weight_decay)``; configurations without an ``optimizer`` node give the upstream Adam without decay."""
    node = cfg.get("optimizer") or {}
    name = str(node.get("name", "adam"))
    weight_decay = float(node.get("weight_decay", 0.0))
    if name not in OPTIMISERS:
        raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMISERS}.")
    if weight_decay < 0 or not math.isfinite(weight_decay):
        raise ValueError("optimizer.weight_decay must be a finite non-negative number.")
    return name, weight_decay


def make_optimiser(cfg: DictConfig, parameters) -> torch.optim.Optimizer:
    """Adam with coupled L2 (``adam``, the upstream choice) or Adam with decoupled weight decay (``adamw``).

    ``torch.optim.AdamW`` multiplies every parameter by ``1 - lr * weight_decay`` before the Adam
    step (Loshchilov and Hutter, 2019), so the decay follows the learning-rate schedule and is not
    rescaled by Adam's second-moment estimate. ``torch.optim.Adam`` with a weight decay adds
    ``weight_decay * parameter`` to the gradient instead, which Adam then normalises, so that
    parameters with large gradient variance are barely regularised. With a weight decay of 0 both
    are the upstream update.
    """
    name, weight_decay = optimiser_settings(cfg)
    cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
    return cls(parameters, betas=(0.9, 0.999), lr=cfg.start_lr, weight_decay=weight_decay)


def early_stopping_settings(cfg: DictConfig) -> dict | None:
    """The validated ``early_stopping`` node as a plain dict, or None when early stopping is not used.

    ``{"caps": [...], "rules": {label: {"patience": int | None, "min_delta": float}}}``: every rule is
    applied with every cap; its models are saved as ``<label>_cap<cap>``.
    """
    node = cfg.get("early_stopping")
    if not node:
        return None
    caps = sorted({int(c) for c in node.caps})
    rules = {}
    for label, rule in node.rules.items():
        patience = rule.get("patience")
        rules[str(label)] = {"patience": None if patience is None else int(patience), "min_delta": float(rule.get("min_delta", 0.0))}
        if rules[label]["patience"] is not None and rules[label]["patience"] < 1:
            raise ValueError("early_stopping patience must be at least one epoch (or null: never stop).")
        if rules[label]["min_delta"] < 0:
            raise ValueError("early_stopping min_delta must be non-negative.")
    if not rules:
        raise ValueError("early_stopping needs at least one rule.")
    if not caps or caps[0] < 1 or caps[-1] > int(cfg.max_epochs):
        raise ValueError("early_stopping.caps must be epochs between 1 and max_epochs.")
    if int(cfg.evaluation.get("validate_every", 1)) != 1:
        raise ValueError("early stopping needs the validation error after every epoch (evaluation.validate_every = 1).")
    return {"caps": caps, "rules": rules}


def snapshot_epochs(cfg: DictConfig) -> list[int]:
    epochs = sorted({int(e) for e in (cfg.get("snapshots") or {}).get("epochs", [])})
    if epochs and (epochs[0] < 1 or epochs[-1] > int(cfg.max_epochs)):
        raise ValueError("snapshots.epochs must lie between 1 and max_epochs.")
    return epochs


def improves(value: float, best: float, min_delta: float) -> bool:
    return value < best * (1.0 - min_delta)


def early_stopping_outcome(validation: list[float], patience: int | None, min_delta: float, cap: int) -> dict:
    """The rule of this module applied to per-epoch validation errors (epoch ``i + 1`` is ``validation[i]``).

    Returns ``best_epoch`` (the selected model), ``stop_epoch`` (None if the rule did not stop
    before the cap) and ``consumed_epochs`` (epochs trained: the stop epoch, or the cap).
    """
    if cap > len(validation):
        raise ValueError(f"cap {cap} beyond the {len(validation)} recorded epochs")
    best, best_epoch = math.inf, None
    for epoch in range(1, cap + 1):
        if improves(validation[epoch - 1], best, min_delta):
            best, best_epoch = validation[epoch - 1], epoch
        if patience is not None and epoch - best_epoch >= patience:
            return {"best_epoch": best_epoch, "stop_epoch": epoch, "consumed_epochs": epoch}
    return {"best_epoch": best_epoch, "stop_epoch": None, "consumed_epochs": cap}


class EarlyStoppingTracker:
    """Follows one rule during training and keeps the weights each cap would select (on the CPU)."""

    def __init__(self, rule: dict, caps: list[int]):
        self.settings = {**rule, "caps": list(caps)}
        self.best = math.inf
        self.best_epoch = None
        self.best_state = None
        self.stop_epoch = None
        self.at_stop = None  # (best_epoch, state) frozen at the stop

    def update(self, epoch: int, value: float, model: torch.nn.Module) -> list[tuple[int, int, int | None, dict]]:
        """Record the validation error of ``epoch``; return ``(cap, selected_epoch, stop_epoch, state)`` for caps reached now."""
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite validation error at epoch {epoch}")
        if improves(value, self.best, self.settings["min_delta"]):
            self.best, self.best_epoch = value, epoch
            # the copy is part of the cost of early stopping: it is what a halted run would keep
            self.best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        patience = self.settings["patience"]
        if self.stop_epoch is None and patience is not None and epoch - self.best_epoch >= patience:
            self.stop_epoch = epoch
            self.at_stop = (self.best_epoch, self.best_state)
        reached = []
        for cap in self.settings["caps"]:
            if cap == epoch:
                selected, state = self.at_stop if self.stop_epoch is not None else (self.best_epoch, self.best_state)
                reached.append((cap, selected, self.stop_epoch, state))
        return reached


SELECTION_FILE = "results/audit/regularisation_selection.json"


def guard_test_evaluation(cfg: DictConfig, root) -> None:
    """Enforce the frozen study's test gate on the standard evaluation entry points."""
    from .freeze import check_restore

    if cfg.name.startswith("reg_cuda_wd") or (cfg.get("evaluation") or {}).get("test_after_selection"):
        check_restore(cfg, root)
