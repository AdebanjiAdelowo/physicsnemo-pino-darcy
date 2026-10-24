"""Error metrics. The relative L2 error is the same in normalised and in physical units."""

import numpy as np
import torch


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error ``||prediction - target||_2 / ||target||_2``.

    Parameters
    ----------
    prediction, target : torch.Tensor
        Tensors of shape ``[batch, ...]``.

    Returns
    -------
    torch.Tensor
        Shape ``[batch]``.
    """
    diff = (prediction - target).flatten(1)
    return torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1)


def summarise(values: torch.Tensor) -> dict:
    """Mean, sample standard deviation, median and maximum of a 1D tensor.

    The median is ``torch.median``: for an even count, the lower of the two middle values,
    not their average. The stored results use this definition.
    """
    values = values.double()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=True).item() if values.numel() > 1 else 0.0,
        "median": values.median().item(),
        "max": values.max().item(),
    }


def paired_seed_interval(differences, confidence: float = 0.95) -> dict:
    """Mean of paired per-seed differences with a Student-t interval over seeds.

    ``differences[i]`` is a model's metric minus the comparison model's metric for training
    seed ``i`` (same data, same test samples). The interval treats seeds as the independent
    units; with ``n`` seeds it has ``n - 1`` degrees of freedom and is wide for small ``n``.
    """
    from scipy import stats

    d = np.asarray(differences, dtype=np.float64)
    n = d.size
    mean = float(d.mean())
    out = {"n_seeds": n, "per_seed": d.tolist(), "mean": mean, "std": None, "ci_low": None, "ci_high": None, "t_p_value": None}
    if n > 1:
        sd = float(d.std(ddof=1))
        half = float(stats.t.ppf(0.5 + confidence / 2, n - 1)) * sd / np.sqrt(n)
        out.update(std=sd, ci_low=mean - half, ci_high=mean + half)
        out["t_p_value"] = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (0.0 if mean != 0 else 1.0)
    out["seeds_negative"] = int((d < 0).sum())
    # exact two-sided sign test; with 3 seeds its smallest attainable value is 0.25
    k = min(out["seeds_negative"], int((d > 0).sum()))
    out["sign_test_p_value"] = float(min(1.0, 2 * stats.binom.cdf(k, int((d != 0).sum()), 0.5))) if (d != 0).any() else 1.0
    return out


def paired_bootstrap(a, b, resamples: int = 10_000, seed: int = 0, confidence: float = 0.95) -> dict:
    """Mean of ``a - b`` over shared test samples with a percentile bootstrap interval.

    Measures test-sample variability for fixed trained models only, as in the audit (F6).
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(d.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
        "fraction_improved": float((d < 0).mean()),
    }
