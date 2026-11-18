from __future__ import annotations

import numpy as np
import pandas as pd


def rmse_masked(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = ~np.isnan(y_true)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def mcrmse(y_true: pd.DataFrame, y_pred: pd.DataFrame, targets: list[str]) -> tuple[float, dict]:
    scores = {t: rmse_masked(y_true[t].values, y_pred[t].values) for t in targets}
    valid = [v for v in scores.values() if not np.isnan(v)]
    return float(np.mean(valid)), scores
