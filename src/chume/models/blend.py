from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def fit_blend(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    m = ~np.isnan(y) & ~np.isnan(preds).any(axis=1)
    w, _ = nnls(preds[m], y[m])
    if w.sum() <= 0:
        w = np.ones(preds.shape[1])
    return w / w.sum()


def apply_blend(preds: np.ndarray, w: np.ndarray) -> np.ndarray:
    return preds @ w


def nested_blend_oof(
    y: np.ndarray, preds: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    out = np.full(len(y), np.nan)
    ws = []
    for k in range(int(folds.max()) + 1):
        tr = folds != k
        w = fit_blend(np.where(tr, y, np.nan), preds)
        ws.append(w)
        va = folds == k
        out[va] = apply_blend(preds[va], w)
    return out, ws
