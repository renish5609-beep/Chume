from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")

import numpy as np
import pandas as pd


def _fit_lgbm(Xtr, ytr, Xva, yva, params, seed):
    import lightgbm as lgb

    p = dict(params)
    n_est = p.pop("n_estimators", 3000)
    es = p.pop("early_stopping_rounds", 150)
    p["seed"] = seed
    m = lgb.LGBMRegressor(n_estimators=n_est, **p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(es, verbose=False)])
    return m


def _fit_cat(Xtr, ytr, Xva, yva, params, seed):
    from catboost import CatBoostRegressor

    p = dict(params)
    p["random_seed"] = seed
    m = CatBoostRegressor(**p)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    return m


FITTERS = {"lgbm": _fit_lgbm, "catboost": _fit_cat}


def fit_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: pd.DataFrame,
    kind: str,
    params: dict,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, list]:
    obs = ~np.isnan(y)
    K = int(folds.max()) + 1
    oof = np.full(len(y), np.nan)
    test_pred = np.zeros(len(X_test))
    models = []
    fit = FITTERS[kind]
    for k in range(K):
        va = (folds == k) & obs
        inner = (folds == (k + 1) % K) & obs
        tr = (folds != k) & (folds != (k + 1) % K) & obs
        if va.sum() == 0 or tr.sum() < 20 or inner.sum() == 0:
            continue
        pv = np.zeros(va.sum())
        for s in seeds:
            m = fit(X[tr], y[tr], X[inner], y[inner], params, s)
            pv += m.predict(X[va]) / len(seeds)
            test_pred += m.predict(X_test) / (len(seeds) * K)
            models.append(m)
        oof[va] = pv
    return oof, test_pred, models


def importance(models: list, cols: list[str]) -> pd.Series:
    imp = np.zeros(len(cols))
    for m in models:
        if hasattr(m, "feature_importances_"):
            v = np.asarray(m.feature_importances_, dtype=float)
            imp += v / (v.sum() + 1e-12)
    return pd.Series(imp / max(len(models), 1), index=cols).sort_values(ascending=False)
