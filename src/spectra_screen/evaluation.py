"""Shared repeated-k-fold evaluation used by benchmarks and the CLI.

One entry point so every number in the README comes from the same code path:
5-fold x 5 repeats, fixed seed, per-fold metric reported as mean +/- std.
"""

import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def evaluate(X, y, model_name, task="regression", n_splits=5, n_repeats=5, random_state=42,
             scale=False):
    """Repeated k-fold evaluation. `scale=True` fits a StandardScaler on the
    training fold only (no leakage), for datasets without row-wise preprocessing."""
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    from spectra_screen.models import get_model

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    if task == "regression":
        splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    fold_scores = []
    pooled_true, pooled_pred = [], []
    for train_idx, test_idx in splitter.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        if scale:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler().fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        model = get_model(model_name, task=task, random_state=random_state, n_features=X.shape[1])
        model.fit(Xtr, y[train_idx])
        pred = np.asarray(model.predict(Xte)).ravel()
        if task == "regression":
            fold_scores.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        else:
            fold_scores.append(float(accuracy_score(y[test_idx], pred)))
        pooled_true.append(y[test_idx])
        pooled_pred.append(pred)

    pooled_true = np.concatenate(pooled_true)
    pooled_pred = np.concatenate(pooled_pred)

    result = {
        "model": model_name,
        "task": task,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "metric": "rmse" if task == "regression" else "accuracy",
        "fold_scores_mean": float(np.mean(fold_scores)),
        "fold_scores_std": float(np.std(fold_scores)),
    }
    if task == "regression":
        result["pooled_r2"] = float(r2_score(pooled_true, pooled_pred))
    else:
        result["pooled_accuracy"] = float(accuracy_score(pooled_true, pooled_pred))
    return result
