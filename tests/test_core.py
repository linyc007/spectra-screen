"""Offline tests with synthetic spectra: no network, deterministic."""

import numpy as np
import pytest

from spectra_screen.evaluation import evaluate
from spectra_screen.preprocess import savgol, snv
from spectra_screen.screening import discrete_screen, select_top_k, vip_scores


def make_spectra(n=120, n_bands=60, seed=0):
    """Synthetic NIR-like data: two absorptions + noise; y tracks peak height."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(850, 1050, n_bands)
    peaks = 0.8 * np.exp(-0.5 * ((wl - 930) / 8) ** 2) + 0.5 * np.exp(-0.5 * ((wl - 1010) / 12) ** 2)
    X = rng.normal(1.0, 0.05, size=(n, n_bands)) + peaks[None, :] * rng.normal(0, 1, size=(n, 1))
    peak_idx = int(np.argmin(np.abs(wl - 930)))
    y = X[:, peak_idx] * 10 + rng.normal(0, 0.1, n)
    return X, y, wl


def test_snv_normalizes_rows():
    X, _, _ = make_spectra()
    Z = snv(X)
    np.testing.assert_allclose(Z.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(Z.std(axis=1), 1.0, atol=1e-10)


def test_savgol_rejects_bad_params():
    X, _, _ = make_spectra(n_bands=30)
    with pytest.raises(ValueError):
        savgol(X, window=10)  # even window
    with pytest.raises(ValueError):
        savgol(X, window=31)  # window > n_bands


def test_vip_peaks_score_highest():
    X, y, wl = make_spectra()
    scores = vip_scores(snv(X), y)
    peak_idx = int(np.argmax(scores))
    assert abs(wl[peak_idx] - 930) < 20, f"peak band should dominate, got {wl[peak_idx]}nm"


def test_discrete_enforces_separation():
    X, y, _ = make_spectra()
    sel, corr = discrete_screen(X, y, k=3, min_separation=15)
    assert len(sel) == 3
    diffs = np.diff(sel)
    assert (diffs >= 15).all(), f"bands closer than separation: {sel}"
    # selected bands must be among the highest-correlated ones
    top = set(np.argsort(corr)[::-1][:20])
    assert top.issuperset(set(sel.tolist())) or len(top & set(sel.tolist())) >= 2


def test_select_top_k_sorted():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    sel = select_top_k(scores, k=2)
    assert sel.tolist() == [1, 3]  # top-2 by score, sorted by index


def test_evaluate_regression_smoke():
    X, y, _ = make_spectra()
    res = evaluate(X, y, "ridge", task="regression", n_splits=3, n_repeats=2)
    assert res["metric"] == "rmse"
    assert res["fold_scores_mean"] > 0
    assert -1.0 <= res["pooled_r2"] <= 1.0
