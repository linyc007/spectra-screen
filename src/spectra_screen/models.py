"""Model registry shared by benchmarks and the CLI."""

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVC, SVR


def _tabpfn_regressor():
    """TabPFN v2 regressor, imported lazily (optional dependency).

    Install with `pip install -e '.[tabpfn]'`. Weights cache under
    ~/.cache/tabpfn; on machines without the license token, copying a
    previously-downloaded checkpoint there keeps inference fully offline.
    """
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor()


def get_model(name, task="regression", random_state=0, n_features=None):
    """Return an unfitted estimator by name.

    Regression: pls, ridge, svr. Classification: svm, rf.
    `n_features` clamps PLS components when screening keeps few bands.
    """
    if task == "regression":
        n_comp = min(10, n_features) if n_features else 10
        registry = {
            "pls": lambda: PLSRegression(n_components=max(1, n_comp)),
            "ridge": lambda: Ridge(alpha=1.0),
            "svr": lambda: SVR(C=10.0, epsilon=0.1),
            "tabpfn": _tabpfn_regressor,
        }
    elif task == "classification":
        registry = {
            "svm": lambda: SVC(kernel="rbf", C=10.0, random_state=random_state),
            "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=random_state),
        }
    else:
        raise ValueError(f"unknown task '{task}' (use 'regression' or 'classification')")
    try:
        return registry[name]()
    except KeyError:
        raise ValueError(
            f"unknown model '{name}' for task '{task}'; options: {sorted(registry)}"
        ) from None
