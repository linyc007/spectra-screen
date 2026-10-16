"""Dataset loaders. Datasets download on first use and cache under ./data
(or $SPECTRA_SCREEN_DATA). Nothing under data/ is committed."""

import os

import numpy as np

DATA_DIR = os.environ.get("SPECTRA_SCREEN_DATA", os.path.join(".", "data"))


def _data_home():
    path = os.path.abspath(DATA_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def tecator_wavelengths():
    """Tecator NIR grid: 100 channels at 2 nm steps, 850-1048 nm (nominal 850-1050)."""
    return np.arange(850.0, 1050.0, 2.0)


def load_tecator():
    """OpenML id=505 (Tecator): 240 meat samples, 100 NIR absorbance channels,
    targets moisture / fat / protein (%). Returns (X, y, wavelengths, meta).

    Default task: predict fat content (regression).
    """
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=505, as_frame=True, parser="auto", data_home=_data_home())
    df = ds.frame
    band_cols = [c for c in df.columns if c.startswith("absorbance_")]
    band_cols.sort(key=lambda c: int(c.split("_")[1]))
    X = df[band_cols].to_numpy(dtype=float)
    wavelengths = tecator_wavelengths()
    if len(band_cols) != len(wavelengths):
        raise ValueError(
            f"expected {len(wavelengths)} bands, OpenML returned {len(band_cols)}"
        )
    meta = {
        "name": "tecator",
        "task": "regression",
        "n_samples": X.shape[0],
        "source": "OpenML data_id=505 (origin: Tecator, 850-1050 nm NIR)",
        "aux_targets": {c: df[c].tolist() for c in ("moisture", "protein")},
    }
    return X, df["fat"].to_numpy(dtype=float), wavelengths, meta


def load_indian_pines(min_pixels_per_class=200, random_state=42):
    """Indian Pines AVIRIS scene (145x145 px, 200 bands after water-absorption
    removal), 16 crop/forest classes in the ground-truth mask.

    Downloaded from the HuggingFace mirror (the original EHU host returns 403
    since 2026-10) and cached under the data home. Returns (X, y, meta) where
    rows are labelled pixels only (mask > 0):

    - Class balancing: at most `min_pixels_per_class` rows per class, drawn
      deterministically, so the biggest classes don't swamp the benchmark.
    - No wavelengths returned: AVIRIS band-to-nanometer mapping needs
      calibration metadata the mirror does not ship, so bands are numbered.

    Default task: crop classification.
    """
    from scipy.io import loadmat

    base = "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main"
    data_path = os.path.join(_data_home(), "Indian_pines_corrected.mat")
    gt_path = os.path.join(_data_home(), "Indian_pines_gt.mat")
    for url, path in ((f"{base}/Indian_pines_corrected.mat", data_path),
                      (f"{base}/Indian_pines_gt.mat", gt_path)):
        if not os.path.exists(path):
            import urllib.request

            urllib.request.urlretrieve(url, path)

    cube = loadmat(data_path)["indian_pines_corrected"].astype(float)
    mask = loadmat(gt_path)["indian_pines_gt"].ravel()
    X_all = cube.reshape(-1, cube.shape[2])
    keep = mask > 0
    X_all, y_all = X_all[keep], mask[keep].astype(int)

    rng = np.random.default_rng(random_state)
    selected = []
    for cls in np.unique(y_all):
        idx = np.flatnonzero(y_all == cls)
        if len(idx) > min_pixels_per_class:
            idx = rng.choice(idx, size=min_pixels_per_class, replace=False)
        selected.append(idx)
    sel = np.sort(np.concatenate(selected))

    X, y = X_all[sel], y_all[sel]
    meta = {
        "name": "indian_pines",
        "task": "classification",
        "n_samples": int(X.shape[0]),
        "n_bands": int(X.shape[1]),
        "n_classes": len(np.unique(y)),
        "source": "HF mirror danaroth/indian_pines (origin: AVIRIS Indian Pines, "
        "145x145 px, 200 bands, 16 classes; EHU host 403 since 2026-10)",
        "sampling": f"<= {min_pixels_per_class} px/class, random_state={random_state}",
    }
    return X, y, meta
