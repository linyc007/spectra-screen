"""Regenerate every Indian Pines number quoted in the README.

Usage:  python3 -u benchmarks/run_indian_pines.py
Output: results/indian_pines.json, results/indian_pines_table.md,
        results/indian_pines_pareto.png

Protocol mirrors run_tecator.py: 5-fold x 5 repeats (stratified), fixed seed,
selection computed once on the full labelled set (consistent with the Tecator
protocol; selection-CV leakage noted in the README). Features are standardized
inside each training fold (no leakage).
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

from spectra_screen.datasets import load_indian_pines
from spectra_screen.evaluation import evaluate
from spectra_screen.screening import anova_f_scores, discrete_screen, select_top_k

plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "sans-serif"]

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

MODELS = ["svm", "rf"]
KS = [5, 8, 15]


def main():
    X, y, meta = load_indian_pines()
    print(
        f"indian_pines: n={meta['n_samples']}, bands={meta['n_bands']}, "
        f"classes={meta['n_classes']} ({meta['sampling']})"
    )

    def selection(method, k):
        if method == "full":
            return np.arange(X.shape[1])
        if method == "anova-topk":
            return select_top_k(anova_f_scores(X, y), k)
        if method == "discrete":
            sep = max(1, min(10, X.shape[1] // k))
            sel, _ = discrete_screen(X, y, k, min_separation=sep, criterion="anova")
            return sel
        raise ValueError(method)

    rows = []
    for method in ["full", "anova-topk", "discrete"]:
        for k in ([None] if method == "full" else KS):
            sel = selection(method, k)
            for model in MODELS:
                res = evaluate(
                    X[:, sel], y, model, task="classification", scale=True
                )
                res.update(method=method, k=(k or X.shape[1]))
                rows.append(res)
                print(
                    f"{method:11s} k={res['k']:>3} {model:3s} "
                    f"ACC={res['fold_scores_mean']:.4f}+/-{res['fold_scores_std']:.4f} "
                    f"pooled={res['pooled_accuracy']:.4f}"
                )

    (RESULTS / "indian_pines.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "| Method | Bands | Model | Accuracy (mean +/- std) | Pooled accuracy |",
        "|---|---|---|---|---|",
    ]
    for method in ["full", "anova-topk", "discrete"]:
        subset = [r for r in rows if r["method"] == method]
        for k in sorted({r["k"] for r in subset}):
            for model in MODELS:
                cell = [r for r in subset if r["k"] == k and r["model"] == model]
                if not cell:
                    continue
                r = cell[0]
                lines.append(
                    f"| {method} | {r['k']} | {model} "
                    f"| {r['fold_scores_mean']:.4f} +/- {r['fold_scores_std']:.4f} "
                    f"| {r['pooled_accuracy']:.4f} |"
                )
    (RESULTS / "indian_pines_table.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"full": "#888888", "anova-topk": "#1f77b4", "discrete": "#d62728"}
    for method in ["anova-topk", "discrete"]:
        pts = sorted(
            (r["k"], r["fold_scores_mean"])
            for r in rows
            if r["method"] == method and r["model"] == "svm"
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=method, color=colors[method])
    full_acc = next(
        r["fold_scores_mean"] for r in rows if r["method"] == "full" and r["model"] == "svm"
    )
    ax.axhline(full_acc, color=colors["full"], ls="--", label=f"full spectrum ({full_acc:.3f})")
    ax.set_xlabel("bands kept")
    ax.set_ylabel("accuracy (16-class crop)")
    ax.set_title("Indian Pines: screening Pareto (SVM-RBF, 5-fold x 5 repeats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "indian_pines_pareto.png", dpi=150)
    print(f"wrote {RESULTS}/indian_pines.json, indian_pines_table.md, indian_pines_pareto.png")


if __name__ == "__main__":
    main()
