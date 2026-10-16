"""Regenerate every Tecator number quoted in the README, plus the Pareto figure.

Usage:  make benchmarks   (or: python3 -u benchmarks/run_tecator.py)
Output: results/tecator.json, results/tecator_table.md, results/tecator_pareto.png
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectra_screen.datasets import load_tecator
from spectra_screen.evaluation import evaluate
from spectra_screen.preprocess import savgol, snv
from spectra_screen.screening import cars, discrete_screen, select_top_k, vip_scores

# Include a CJK-capable fallback in the font list (harmless elsewhere)
plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "sans-serif"]

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODELS = ["pls", "ridge", "svr"]
KS = [5, 8, 15]


def main():
    X, y, _wl, meta = load_tecator()
    X = savgol(snv(X), window=11, polyorder=2)
    print(f"tecator: n={meta['n_samples']}, bands={X.shape[1]}")

    selection_cache = {}

    def selection(method, k):
        key = (method, k)
        if key in selection_cache:
            return selection_cache[key]
        if method == "full":
            sel = np.arange(X.shape[1])
        elif method == "vip":
            sel = select_top_k(vip_scores(X, y), k)
        elif method == "discrete":
            sep = max(1, min(10, X.shape[1] // k))
            sel, _ = discrete_screen(X, y, k, min_separation=sep)
        elif method == "cars":
            sel = cars(X, y, k)
        else:
            raise ValueError(method)
        selection_cache[key] = sel
        return sel

    rows = []
    for method in ["full", "vip", "discrete", "cars"]:
        for k in ([None] if method == "full" else KS):
            sel = selection(method, k)
            for model in MODELS:
                res = evaluate(X[:, sel], y, model, task="regression")
                res.update(method=method, k=(k or X.shape[1]))
                rows.append(res)
                print(
                    f"{method:9s} k={res['k']:>3} {model:5s} "
                    f"RMSE={res['fold_scores_mean']:.3f}+/-{res['fold_scores_std']:.3f} "
                    f"R2={res['pooled_r2']:.4f}"
                )

    (RESULTS / "tecator.json").write_text(json.dumps(rows, indent=2))

    # markdown table: best model per (method, k)
    lines = [
        "| Method | Bands | Best model | RMSE (fat %, mean +/- std) | Pooled R2 |",
        "|---|---|---|---|---|",
    ]
    for method in ["full", "vip", "discrete", "cars"]:
        subset = [r for r in rows if r["method"] == method]
        for k in sorted({r["k"] for r in subset}):
            cell = [r for r in subset if r["k"] == k]
            best = min(cell, key=lambda r: r["fold_scores_mean"])
            lines.append(
                f"| {method} | {best['k']} | {best['model']} "
                f"| {best['fold_scores_mean']:.3f} +/- {best['fold_scores_std']:.3f} "
                f"| {best['pooled_r2']:.4f} |"
            )
    (RESULTS / "tecator_table.md").write_text("\n".join(lines) + "\n")

    # Pareto figure: RMSE vs number of bands
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"full": "#888888", "vip": "#1f77b4", "discrete": "#d62728", "cars": "#2ca02c"}
    for method in ["vip", "discrete", "cars"]:
        pts = sorted(
            [
                (r["k"], min(rr["fold_scores_mean"] for rr in rows if rr["method"] == method and rr["k"] == r["k"]))
                for r in rows
                if r["method"] == method
            ]
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=method, color=colors[method])
    full_rmse = min(r["fold_scores_mean"] for r in rows if r["method"] == "full")
    ax.axhline(full_rmse, color=colors["full"], ls="--", label=f"full spectrum ({full_rmse:.3f})")
    ax.set_xlabel("bands kept")
    ax.set_ylabel("RMSE (fat %)")
    ax.set_title("Tecator: screening Pareto (5-fold x 5 repeats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "tecator_pareto.png", dpi=150)
    print(f"wrote {RESULTS}/tecator.json, tecator_table.md, tecator_pareto.png")


if __name__ == "__main__":
    main()
