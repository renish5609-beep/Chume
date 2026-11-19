from __future__ import annotations

import numpy as np


def fit_gate(y: np.ndarray, pred: np.ndarray, mu: float, w_max: float = 1.2) -> float:
    m = ~np.isnan(y) & ~np.isnan(pred)
    d = pred[m] - mu
    r = y[m] - mu
    denom = float((d * d).sum())
    if denom <= 0:
        return 0.0
    return float(np.clip((d * r).sum() / denom, 0.0, w_max))


def apply_gate(pred: np.ndarray, mu: float, w: float) -> np.ndarray:
    return mu + w * (pred - mu)


def gate_stability(y: np.ndarray, pred: np.ndarray, folds: np.ndarray, mu: float) -> list[float]:
    return [fit_gate(np.where(folds == k, y, np.nan), pred, mu) for k in range(int(folds.max()) + 1)]


def nested_gate_oof(
    y: np.ndarray, pred: np.ndarray, folds: np.ndarray, w_max: float = 1.2
) -> tuple[np.ndarray, list[float]]:
    out = np.full_like(pred, np.nan, dtype=float)
    ws = []
    for k in range(int(folds.max()) + 1):
        tr = (folds != k) & ~np.isnan(y)
        mu = float(np.nanmean(y[tr]))
        w = fit_gate(np.where(tr, y, np.nan), pred, mu, w_max)
        ws.append(w)
        va = folds == k
        out[va] = apply_gate(pred[va], mu, w)
    return out, ws
