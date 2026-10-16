"""Wavelength screening methods: VIP-PLS, CARS, and greedy discrete screening.

Every selector returns the chosen band indices sorted in wavelength order,
plus the full score array used to choose them, so results are auditable.
"""

import numpy as np


def vip_scores(X, y, n_components=10):
    """VIP (variable importance in projection) scores from a PLS model.

    Higher = more important. Reference: classic chemometrics (Wold et al.).
    """
    from sklearn.cross_decomposition import PLSRegression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)

    t = pls.x_scores_  # (n_samples, n_components)
    w = pls.x_weights_  # (n_bands, n_components)
    q = pls.y_loadings_.ravel()  # (n_components,)

    ssy = ((t**2) @ (q**2)).sum()
    vips = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        weight_sq = (w[j, :] ** 2) * ((t**2).sum(axis=0))
        vips[j] = np.sqrt(X.shape[1] * weight_sq.sum() / ssy)
    return vips


def select_top_k(scores, k):
    """Top-k band indices by score, returned sorted in wavelength order."""
    idx = np.argsort(np.asarray(scores))[::-1][:k]
    return np.sort(idx)


def anova_f_scores(X, y):
    """Per-band ANOVA F-statistic: class-separation relevance for
    classification tasks (Pearson correlation with integer class labels is
    not meaningful for >2 classes)."""
    from sklearn.feature_selection import f_classif

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0)


def discrete_screen(X, y, k, min_separation=10, criterion="correlation"):
    """Greedy discrete band screening with enforced wavelength separation.

    1. Rank bands by relevance: |Pearson correlation| with the target
       (criterion="correlation", regression) or ANOVA F-statistic
       (criterion="anova", classification).
    2. Walk down the ranking, accepting a band only if it is at least
       `min_separation` bands away from every already-accepted band.
    3. Stop after k accepted bands.

    Semantics: the result is a set of well-separated "discrete" wavelengths,
    the filter-based seed of the discrete screening method in
    Y. Lin et al., WHISPERS 2019 (simplified, univariate relevance only).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if min_separation < 1:
        raise ValueError("min_separation must be >= 1")

    if criterion == "correlation":
        scores = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) for j in range(X.shape[1])])
        scores = np.nan_to_num(scores, nan=0.0)
    elif criterion == "anova":
        scores = anova_f_scores(X, y)
    elif criterion == "mi-classification":
        from sklearn.feature_selection import mutual_info_classif

        scores = np.nan_to_num(mutual_info_classif(X, y, random_state=0), nan=0.0)
    elif criterion == "mi-regression":
        from sklearn.feature_selection import mutual_info_regression

        scores = np.nan_to_num(mutual_info_regression(X, y, random_state=0), nan=0.0)
    else:
        raise ValueError(
            f"unknown criterion '{criterion}' (use 'correlation', 'anova', "
            "'mi-classification' or 'mi-regression')"
        )

    accepted = []
    for j in np.argsort(scores)[::-1]:
        if all(abs(j - a) >= min_separation for a in accepted):
            accepted.append(int(j))
            if len(accepted) == k:
                break
    if len(accepted) < k:
        raise ValueError(
            f"could only accept {len(accepted)}/{k} bands with min_separation={min_separation}"
        )
    return np.sort(np.array(accepted)), scores


def cars(X, y, k, n_iterations=60, mc_fraction=0.8, random_state=42):
    """Competitive Adaptive Rewarded Sampling (Li et al., Chemom. Intell. Lab.
    Syst. 2009), deterministic seeded variant.

    Each iteration: fit PLS on a Monte Carlo subsample, weight bands by
    |coefficient| x |correlation|, keep an exponentially shrinking share,
    and track CV RMSE of the retained set. The best-performing set wins.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_predict

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    rng = np.random.default_rng(random_state)

    n_bands = X.shape[1]
    candidate = np.arange(n_bands)
    best = (np.inf, candidate.copy())

    decay = (k / n_bands) ** (1.0 / max(n_iterations - 1, 1))
    for i in range(n_iterations):
        n_keep = max(k, round(n_bands * decay**i))

        mc_idx = rng.choice(len(y), size=max(10, int(mc_fraction * len(y))), replace=False)
        pls = PLSRegression(n_components=min(10, len(candidate), len(mc_idx)))
        pls.fit(X[mc_idx][:, candidate], y[mc_idx])
        b = np.abs(pls.coef_.ravel())
        r = np.array(
            [abs(np.corrcoef(X[mc_idx][:, candidate][:, j], y[mc_idx])[0, 1]) for j in range(len(candidate))]
        )
        r = np.nan_to_num(r, nan=0.0)
        weight = b * r

        keep_local = np.argsort(weight)[::-1][:n_keep]
        candidate = candidate[keep_local]

        if len(candidate) >= k:
            pls_cv = PLSRegression(n_components=min(10, candidate.size))
            pred = cross_val_predict(pls_cv, X[:, candidate], y, cv=5)
            rmse = float(np.sqrt(np.mean((y - pred.ravel()) ** 2)))
            if rmse < best[0]:
                best = (rmse, candidate.copy())

    return np.sort(best[1])
