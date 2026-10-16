"""TabPFN comparison on Tecator.

This benchmark needs the `tabpfn` optional extra plus a model-weight cache.
It is excluded from `make benchmarks` because:

- TabPFN download/license requires a token on first use on a new machine;
- on shared HPC login nodes, pip/network setup belongs in a job script.

Run it standalone on a machine where TabPFN works (see HPC notes in the
README):  python3 -u benchmarks/run_tabpfn_tecator.py

Writes results/tabpfn_tecator.json; the README tab is updated from this file.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectra_screen.datasets import load_tecator
from spectra_screen.evaluation import evaluate
from spectra_screen.preprocess import savgol, snv
from spectra_screen.screening import cars, discrete_screen, select_top_k, vip_scores

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

KS = [5, 8, 15]


def main():
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        raise SystemExit("tabpfn not installed: pip install -e '.[tabpfn]'")

    X, y, _wl, meta = load_tecator()
    X = savgol(snv(X), window=11, polyorder=2)
    print(f"tecator: n={meta['n_samples']}, bands={X.shape[1]}", flush=True)

    rows = []

    def run(method, k, sel):
        res = evaluate(X[:, sel], y, "tabpfn", task="regression")
        res.update(method=method, k=k)
        rows.append(res)
        print(
            f"{method:9s} k={k:>3} tabpfn "
            f"RMSE={res['fold_scores_mean']:.3f}+/-{res['fold_scores_std']:.3f} "
            f"R2={res['pooled_r2']:.4f}",
            flush=True,
        )

    run("full", X.shape[1], list(range(X.shape[1])))
    for k in KS:
        run("cars", k, cars(X, y, k))
    for k in KS:
        sel, _ = discrete_screen(X, y, k, min_separation=max(1, min(10, X.shape[1] // k)))
        run("discrete", k, sel)
    for k in KS:
        run("vip", k, select_top_k(vip_scores(X, y), k))

    out = RESULTS / "tabpfn_tecator.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
