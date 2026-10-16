"""Command line interface.

Examples:
    screener tecator --method discrete --k 8 --model pls
    screener tecator --method none --model pls          # full-spectrum baseline
"""

import argparse


def _run(args):
    from spectra_screen.datasets import load_tecator
    from spectra_screen.evaluation import evaluate
    from spectra_screen.preprocess import savgol, snv
    from spectra_screen.screening import cars, discrete_screen, select_top_k, vip_scores

    X, y, wl, meta = load_tecator()
    X = savgol(snv(X), window=11, polyorder=2)

    if args.method == "none":
        sel = list(range(X.shape[1]))
        label = "full spectrum"
    elif args.method == "vip":
        sel = select_top_k(vip_scores(X, y), args.k)
        label = f"VIP top-{args.k}"
    elif args.method == "discrete":
        sel, _ = discrete_screen(X, y, args.k, min_separation=args.min_separation)
        label = f"discrete top-{args.k} (sep={args.min_separation})"
    elif args.method == "cars":
        sel = cars(X, y, args.k)
        label = f"CARS top-{args.k}"
    else:
        raise SystemExit(f"unknown method: {args.method}")

    res = evaluate(X[:, sel], y, args.model, task="regression")
    bands = ", ".join(f"{wl[j]:.0f}" for j in sel[:10]) + (" ..." if len(sel) > 10 else "")
    print(f"dataset      : {meta['name']} (n={meta['n_samples']}, {X.shape[1]} bands)")
    print(f"method       : {label}")
    print(f"bands        : {bands}")
    print(f"model        : {args.model}")
    print(
        f"RMSE (fat %) : {res['fold_scores_mean']:.3f} +/- {res['fold_scores_std']:.3f}"
        f"  (5-fold x 5 repeats)"
    )
    print(f"pooled R2    : {res['pooled_r2']:.4f}")
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(prog="screener", description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)

    tec = sub.add_parser("tecator", help="Tecator NIR fat-regression benchmark")
    tec.add_argument("--method", default="discrete", choices=["none", "vip", "discrete", "cars"])
    tec.add_argument("--k", type=int, default=8, help="number of bands to keep")
    tec.add_argument("--model", default="pls", choices=["pls", "ridge", "svr"])
    tec.add_argument("--min-separation", type=int, default=10)
    tec.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
