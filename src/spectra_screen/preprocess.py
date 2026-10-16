"""Spectral preprocessing: row-wise normalization and smoothing."""

import numpy as np
from scipy.signal import savgol_filter


def snv(X):
    """Standard normal variate: normalize each spectrum to zero mean, unit std.

    Removes multiplicative scatter offsets common in NIR reflectance data.
    """
    X = np.asarray(X, dtype=float)
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def savgol(X, window=11, polyorder=2, deriv=0):
    """Savitzky-Golay smoothing/derivative along the wavelength axis."""
    X = np.asarray(X, dtype=float)
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if window <= polyorder:
        raise ValueError("window must be larger than polyorder")
    if X.shape[1] <= window:
        raise ValueError(f"n_bands={X.shape[1]} must exceed window={window}")
    return savgol_filter(X, window, polyorder, deriv=deriv, axis=1)
