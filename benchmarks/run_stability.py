"""Stability analysis: KSC forestry scene + Tecator LASSO stability.

Usage: python3 -u benchmarks/run_stability.py
Output: results/stability_ksc.json, results/stability_tecator.json,
        reports/ figures + reports/band_stability_sample_report.md
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

from spectra_screen.datasets import load_kennedy_space_center, load_tecator
from spectra_screen.evaluation import evaluate
from spectra_screen.preprocess import savgol, snv
from spectra_screen.screening import cars, discrete_screen
from spectra_screen.stability import (
    consensus_bands,
    cv_selection_repeatability,
    lasso_selection_frequency,
)

plt.rcParams["font.family"] = ["Arial Unicode MS", "Hiragino Sans GB", "sans-serif"]

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
RESULTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def part_ksc():
    X, y, meta = load_kennedy_space_center()
    print(
        f"KSC: n={meta['n_samples']}, bands={meta['n_bands']}, "
        f"classes={meta['n_classes']} ({meta['sampling']})",
        flush=True,
    )

    k = 15
    rep_disc = cv_selection_repeatability(
        X, y, k, method="discrete", task="classification",
        n_splits=5, n_repeats=10, min_separation=8, random_state=42,
    )
    rep_anova = cv_selection_repeatability(
        X, y, k, method="anova-topk", task="classification",
        n_splits=5, n_repeats=10, random_state=42,
    )
    print(
        f"  discrete-15: Jaccard={rep_disc['jaccard_mean']:.3f} | "
        f"anova-top15: Jaccard={rep_anova['jaccard_mean']:.3f}",
        flush=True,
    )

    # consensus sets -> accuracy
    cons_disc = consensus_bands(rep_disc["frequency"], k, threshold=0.6)
    cons_anova = consensus_bands(rep_anova["frequency"], k, threshold=0.6)
    acc_cons_disc = evaluate(X[:, cons_disc], y, "svm", task="classification", scale=True)
    acc_cons_anova = evaluate(X[:, cons_anova], y, "svm", task="classification", scale=True)
    acc_single = evaluate(
        X[:, discrete_screen(X, y, k, min_separation=8, criterion="anova")[0]],
        y, "svm", task="classification", scale=True,
    )
    print(
        f"  acc consensus-discrete={acc_cons_disc['fold_scores_mean']:.4f} "
        f"consensus-anova={acc_cons_anova['fold_scores_mean']:.4f} "
        f"single-shot={acc_single['fold_scores_mean']:.4f}",
        flush=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for ax, (rep, label, color) in zip(
        axes,
        [(rep_disc, "discrete (spaced)", "#d62728"), (rep_anova, "ANOVA top-k", "#1f77b4")],
    ):
        ax.bar(np.arange(X.shape[1]), rep["frequency"], width=1.0, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.6, color="#888888", ls=":", lw=0.8)
    axes[1].set_xlabel("AVIRIS band index (KSC, 176 bands)")
    fig.suptitle("KSC: band selection frequency across 50 CV folds (k=15)", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "ksc_stability_heatmap.png", dpi=150)
    plt.close(fig)

    out = {
        "k": k,
        "discrete": {
            "jaccard_mean": rep_disc["jaccard_mean"],
            "consensus_bands": cons_disc.tolist(),
            "acc_consensus": acc_cons_disc,
            "freq_top": sorted(
                [{"band": int(i), "f": float(rep_disc["frequency"][i])}
                 for i in np.argsort(rep_disc["frequency"])[::-1][:20]],
                key=lambda d: -d["f"],
            ),
        },
        "anova_topk": {
            "jaccard_mean": rep_anova["jaccard_mean"],
            "consensus_bands": cons_anova.tolist(),
            "acc_consensus": acc_cons_anova,
        },
        "single_shot_acc": acc_single,
    }
    (RESULTS / "stability_ksc.json").write_text(json.dumps(out, indent=2))
    print("wrote results/stability_ksc.json", flush=True)
    return out


def part_tecator():
    X, y, _wl, meta = load_tecator()
    X = savgol(snv(X), window=11, polyorder=2)
    print(f"tecator: n={meta['n_samples']}, bands={X.shape[1]}", flush=True)

    # Fold-repeat stability for the selection methods that ship in this repo
    # (LASSO-BIC itself collapses to dense solutions on n=240 highly
    # collinear spectra -- a known failure of BIC model selection there --
    # so we report it as a diagnostic, not a selector).
    rep_disc = cv_selection_repeatability(
        X, y, 8, method="discrete", task="regression",
        n_splits=5, n_repeats=10, min_separation=10, random_state=42,
    )
    rep_vip = cv_selection_repeatability(
        X, y, 8, method="vip", task="regression",
        n_splits=5, n_repeats=10, random_state=42,
    )
    freq = lasso_selection_frequency(X, y, n_bootstrap=100, subsample_fraction=0.5,
                                     random_state=42)
    print(
        f"  fold-repeatability Jaccard: discrete={rep_disc['jaccard_mean']:.3f} "
        f"vip={rep_vip['jaccard_mean']:.3f}",
        flush=True,
    )

    cars5 = cars(X, y, 5)
    acc_cars5 = evaluate(X[:, cars5], y, "pls", task="regression")
    acc_full = evaluate(X, y, "pls", task="regression")

    fig, axes = plt.subplots(2, 1, figsize=(10, 5.0), sharex=True)
    for ax, (rep, label, color) in zip(
        axes,
        [(rep_disc, "discrete-8 (spaced)", "#d62728"), (rep_vip, "VIP top-8", "#1f77b4")],
    ):
        ax.bar(np.arange(X.shape[1]), rep["frequency"], width=1.0, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.6, color="#888888", ls=":", lw=0.8)
    axes[1].set_xlabel("Tecator band index (100 NIR channels, 850-1048 nm)")
    fig.suptitle("Tecator: selection frequency across 50 CV folds (k=8)", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "tecator_stability_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.bar(np.arange(X.shape[1]), freq, width=1.0, color="#2ca02c")
    ax.set_xlabel("Tecator band index (100 NIR channels, 850-1048 nm)")
    ax.set_ylabel("LASSO-BIC selection\nfrequency", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Tecator: LASSO-BIC stability selection over 100 bootstrap subsamples "
        "(diagnostic: collapses to dense solutions on collinear NIR)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(REPORTS / "tecator_lasso_stability.png", dpi=150)
    plt.close(fig)

    out = {
        "fold_repeatability_jaccard": {
            "discrete8": rep_disc["jaccard_mean"],
            "vip8": rep_vip["jaccard_mean"],
        },
        "discrete8_freq_top": sorted(
            [{"band": int(i), "f": float(rep_disc["frequency"][i])}
             for i in np.argsort(rep_disc["frequency"])[::-1][:10]],
            key=lambda d: -d["f"],
        ),
        "lasso_diagnostic": {
            "note": "LASSO-BIC selects ~80 of 100 bands on subsamples; not "
            "viable as a selector on n=240 collinear spectra.",
            "top_freq": sorted(
                [{"band": int(i), "f": float(freq[i])}
                 for i in np.argsort(freq)[::-1][:10]],
                key=lambda d: -d["f"],
            ),
        },
        "acc_cars5": acc_cars5,
        "acc_full": acc_full,
    }
    (RESULTS / "stability_tecator.json").write_text(json.dumps(out, indent=2))
    print("wrote results/stability_tecator.json", flush=True)
    return out


if __name__ == "__main__":
    ksc = part_ksc()
    tec = part_tecator()
    print(json.dumps({
        "ksc_jaccard_discrete": ksc["discrete"]["jaccard_mean"],
        "ksc_jaccard_anova": ksc["anova_topk"]["jaccard_mean"],
        "tec_jaccard_discrete8": tec["fold_repeatability_jaccard"]["discrete8"],
        "tec_jaccard_vip8": tec["fold_repeatability_jaccard"]["vip8"],
    }, indent=2))
