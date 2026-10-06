"""Check what the training data represent, before any model is trained.

    python scripts/verify_data.py

Reads the prepared splits and writes ``results/data_check.json`` and ``figures/data_check.png``.

1. Content: keys, value ranges, the relation of ``Kcoeff`` (model input) to ``coeff``.
2. Grid: the reference pressure next to the wall and the scaling of its discrete Laplacian
   decide between a node-centred grid of spacing 1/240 and a cell-centred one of spacing 1/241.
3. Forcing: -k laplace(u) where the permeability is locally constant.
4. Residual of the reference pressure under the upstream PhysicsInformer, for the forcing and
   spacing of the upstream script (``shipped``) and for those of the data (``consistent``).
5. Residual of the reference pressure under several five-point discretisations, in physical units.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from _bootstrap import ROOT
from pino_darcy import data as D
from pino_darcy import engine
from pino_darcy import physics as P
from pino_darcy.config import load_config
from pino_darcy.provenance import collect_metadata, write_json


def neighbours(a):
    return a[:, 2:, 1:-1], a[:, :-2, 1:-1], a[:, 1:-1, 2:], a[:, 1:-1, :-2], a[:, 1:-1, 1:-1]


def split_stats(r, constant):
    a = np.abs(r)
    return {
        "abs_mean": float(a.mean()),
        "abs_mean_constant_k": float(a[constant].mean()),
        "abs_mean_varying_k": float(a[~constant].mean()),
        "abs_median_varying_k": float(np.median(a[~constant])),
    }


def discrete_operators(k, u, h):
    """Residual of -div(k grad u) = 1 for three five-point discretisations, float64."""
    kE, kW, kN, kS, kC = neighbours(k)
    uE, uW, uN, uS, uC = neighbours(u)
    constant = (kE == kC) & (kW == kC) & (kN == kC) & (kS == kC)
    harmonic = lambda a, b: 2 * a * b / (a + b)  # noqa: E731
    expanded = -((kE - kW) * (uE - uW) + (kN - kS) * (uN - uS)) / (4 * h * h) - kC * (uE + uW + uN + uS - 4 * uC) / h**2 - 1
    arithmetic = -((kE + kC) / 2 * (uE - uC) - (kW + kC) / 2 * (uC - uW) + (kN + kC) / 2 * (uN - uC) - (kS + kC) / 2 * (uC - uS)) / h**2 - 1
    harmonic_r = -(harmonic(kE, kC) * (uE - uC) - harmonic(kW, kC) * (uC - uW) + harmonic(kN, kC) * (uN - uC) - harmonic(kS, kC) * (uC - uS)) / h**2 - 1
    return {
        "constant_k_fraction": float(constant.mean()),
        "expanded_central": split_stats(expanded, constant),
        "conservative_arithmetic_mean": split_stats(arithmetic, constant),
        "conservative_harmonic_mean": split_stats(harmonic_r, constant),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", type=Path, default=ROOT, help="directory that receives results/ and figures/")
    args = parser.parse_args()
    cfg = load_config("official")
    manifest = D.prepare(cfg.data, ROOT)
    data_dir = ROOT / cfg.data.dir

    raw = {key: [] for key in ("Kcoeff", "coeff", "sol")}
    for split in ("train", "validation", "test"):
        with h5py.File(data_dir / f"{split}.hdf5") as f:
            for key in raw:
                raw[key].append(np.asarray(f[key][:, 0], dtype=np.float64))
    K, coeff, sol = (np.concatenate(raw[key]) for key in ("Kcoeff", "coeff", "sol"))

    out = {"environment": collect_metadata(torch.device("cpu")), "dataset": manifest, "samples": len(K)}
    out["content"] = {
        "coeff_values": np.unique(coeff).tolist(),
        "Kcoeff_min_max": [float(K.min()), float(K.max())],
        "Kcoeff_equals_coeff_fraction": float((np.abs(K - coeff) < 1e-5).mean()),
        "sol_min_max": [float(sol.min()), float(sol.max())],
        "coeff_std": float(coeff.std()),
        "sol_std": float(sol.std()),
        "upstream_scales": {"K_SCALE": D.K_SCALE, "U_SCALE": D.U_SCALE},
    }

    # A pressure that vanishes on the wall has the value ratio 3 between the second and the
    # first row of a cell-centred grid (distances 3h/2 and h/2), and a zero first row on a
    # node-centred grid.
    edges = [(sol[:, 0], sol[:, 1]), (sol[:, -1], sol[:, -2]), (sol[:, :, 0], sol[:, :, 1]), (sol[:, :, -1], sol[:, :, -2])]
    out["grid"] = {
        "first_row_mean": float(np.mean([a.mean() for a, _ in edges])),
        "second_to_first_row_ratio": float(np.mean([b.mean() / a.mean() for a, b in edges])),
        "max_abs_on_outer_rows": float(max(np.abs(a).max() for a, _ in edges)),
    }

    kE, kW, kN, kS, kC = neighbours(K)
    uE, uW, uN, uS, uC = neighbours(sol)
    constant = (kE == kC) & (kW == kC) & (kN == kC) & (kS == kC)
    out["forcing"] = {}
    for label, h in (("spacing_1_240", 1 / 240), ("spacing_1_241", 1 / 241)):
        f = (-kC * (uE + uW + uN + uS - 4 * uC) / h**2)[constant]
        out["forcing"][label] = {"mean": float(f.mean()), "std": float(f.std())}
    out["forcing"]["normalised_forcing_of_the_data"] = 1 / (D.K_SCALE * D.U_SCALE)
    out["forcing"]["normalised_forcing_in_upstream_script"] = P.RESIDUALS["shipped"]["forcing"]

    sets = {split: D.load_split(data_dir / f"{split}.hdf5") for split in ("train", "validation", "test")}
    informers = engine.make_informers(torch.device("cpu"))
    out["reference_residual_normalised"] = {
        split: {name: P.residual_statistics(engine.residual_fields(inf, s["u"], s["k"]), s["k"]) for name, inf in informers.items()}
        for split, s in sets.items()
    }
    out["discrete_operators_physical_units"] = {
        "note": "residual of -div(k grad u) = 1, spacing 1/241, float64 arithmetic on the float32 data",
        "Kcoeff": discrete_operators(K, sol, 1 / 241),
        "coeff": discrete_operators(coeff, sol, 1 / 241),
    }
    write_json(args.out_root / "results" / "data_check.json", out)

    from pino_darcy.plotting import plot_data_check

    i = 0
    k, u = sets["test"]["k"][i : i + 1], sets["test"]["u"][i : i + 1]
    fields = {name: engine.residual_fields(inf, u, k)[0, 0].numpy() for name, inf in informers.items()}
    print(plot_data_check(k[0, 0].numpy(), u[0, 0].numpy(), fields, args.out_root / "figures" / "data_check.png"))

    c, g, f = out["content"], out["grid"], out["forcing"]
    print(f"samples checked: {len(K)}; coeff values {c['coeff_values']}; Kcoeff equals coeff on {c['Kcoeff_equals_coeff_fraction']:.1%} of cells")
    print(f"second/first row ratio {g['second_to_first_row_ratio']:.3f} (cell-centred grid: 3)")
    print(f"-k laplace(u), constant-k cells: {f['spacing_1_240']['mean']:.5f} with h = 1/240, {f['spacing_1_241']['mean']:.5f} with h = 1/241")
    for name in informers:
        s = out["reference_residual_normalised"]["test"][name]
        print(f"reference residual, {name:>10}: L1 {s['l1']:.3f}, constant-k cells {s['abs_mean_smooth']:.4f}, other cells {s['abs_mean_interface']:.2f}")


if __name__ == "__main__":
    main()
