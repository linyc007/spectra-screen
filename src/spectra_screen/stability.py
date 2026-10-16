"""Band-selection stability analysis.

Two complementary questions for anyone deploying a screened band set:

1. `lasso_selection_frequency` — stability selection (Meinshausen &
   Buhlmann 2010, BIC-alpha variant): how often does each band survive a
   Lasso fit on random subsamples? For regression-type targets.
2. `cv_selection_repeatability` — if we re-run a screening method on each
   CV training fold, do we keep choosing the same bands? Reports per-band
   selection frequency and the mean pairwise Jaccard of the selected sets.

Both return plain dicts/arrays so the benchmark scripts can render reports.
"""

import numpy as np


def lasso_selection_frequency(X, y, n_bootstrap=100, subsample_fraction=0.5,
                              random_state=42):
    """Per-band Lasso selection frequency over bootstrap subsamples.

    Each draw: subsample `subsample_fraction` of rows without replacement,
    standardize features on the subsample, fit LassoLarsIC (BIC-chosen
    alpha), record nonzero coefficients. `y` must be continuous (regression).
    Returns an array of frequencies in [0, 1].
    """
    from sklearn.linear_model import LassoLarsIC
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(random_state)

    n = X.shape[0]
    n_sub = max(10, int(subsample_fraction * n))
    freq = np.zeros(X.shape[1])

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n_sub, replace=False)
        Xs, ys = X[idx], y[idx]
        scaler = StandardScaler().fit(Xs)
        ys_centered = ys - ys.mean()
        lars = LassoLarsIC(criterion="bic")
        lars.fit(scaler.transform(Xs), ys_centered)
        freq[np.nonzero(lars.coef_)[0]] += 1.0

    return freq / float(n_bootstrap)


def _pairwise_jaccard(sets):
    n = len(sets)
    if n < 2:
        return np.ones((max(n, 1), max(n, 1)))
    j = np.ones((n, n))
    for i in range(n):
        for jdx in range(i + 1, n):
            union = len(sets[i] | sets[jdx])
            val = len(sets[i] & sets[jdx]) / union if union else 1.0
            j[i, jdx] = j[jdx, i] = val
    return j


def cv_selection_repeatability(X, y, k, method="discrete", task="classification",
                               n_splits=5, n_repeats=10, min_separation=10,
                               random_state=42):
    """Re-run a screening method on every CV training fold; measure how
    consistently the same bands are chosen.

    Returns dict with:
    - `frequency`: per-band selection frequency in [0, 1] (over all folds)
    - `selected_sets`: the per-fold band index lists
    - `jaccard_mean`: mean pairwise Jaccard between fold-selected sets
    - `jaccard_matrix`: full pairwise matrix
    - `n_folds`
    """
    from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if task == "classification":
        splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                           random_state=random_state)
    else:
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=random_state)

    from spectra_screen.screening import anova_f_scores, discrete_screen, select_top_k, vip_scores

    freq = np.zeros(X.shape[1])
    selected_sets = []

    for train_idx, _ in splitter.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        if method == "discrete":
            criterion = "anova" if task == "classification" else "correlation"
            sep = max(1, min(min_separation, X.shape[1] // k))
            sel, _ = discrete_screen(Xtr, ytr, k, min_separation=sep, criterion=criterion)
        elif method == "anova-topk":
            sel = select_top_k(anova_f_scores(Xtr, ytr), k)
        elif method == "vip":
            sel = select_top_k(vip_scores(Xtr, ytr), k)
        else:
            raise ValueError(f"unknown method '{method}'")
        sel_list = [int(j) for j in sel]
        freq[sel_list] += 1.0
        selected_sets.append(set(sel_list))

    j = _pairwise_jaccard(selected_sets)
    off_diag = j[np.triu_indices(len(selected_sets), k=1)]
    return {
        "frequency": freq / float(len(selected_sets)),
        "selected_sets": selected_sets,
        "jaccard_mean": float(off_diag.mean()) if off_diag.size else 1.0,
        "jaccard_matrix": j,
        "n_folds": len(selected_sets),
    }


def consensus_bands(frequency, k, threshold=0.6):
    """Bands selected in >= `threshold` of folds, ordered by frequency; if
    fewer than k qualify, fill up with the highest-frequency bands."""
    order = np.argsort(frequency)[::-1]
    strong = [int(j) for j in order if frequency[j] >= threshold]
    if len(strong) < k:
        strong = [int(j) for j in order[:k]]
    return np.sort(np.array(strong[:k]))
