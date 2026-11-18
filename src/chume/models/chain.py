from __future__ import annotations

import numpy as np
import pandas as pd

from .gbdt import FITTERS, fit_oof


def _relabel(f: np.ndarray) -> np.ndarray:
    u = np.unique(f)
    m = {v: i for i, v in enumerate(u)}
    return np.array([m[v] for v in f])


def strict_chain(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    folds: np.ndarray,
    X_test: pd.DataFrame,
    params: dict,
    seed: int,
    include_self: bool = False,
    kind: str = "lgbm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(Y.columns)
    K = int(folds.max()) + 1
    oof = np.full((len(X), len(targets)), np.nan)
    test = np.zeros((len(X_test), len(targets)))
    fit = FITTERS[kind]
    for k in range(K):
        tr = folds != k
        va = folds == k
        f_in = _relabel(folds[tr])
        Xtr, Xva = X[tr].reset_index(drop=True), X[va].reset_index(drop=True)
        F_tr = pd.DataFrame(index=Xtr.index)
        F_va = pd.DataFrame(index=Xva.index)
        F_te = pd.DataFrame(index=X_test.index)
        for t in targets:
            ytr = Y[t].values[tr]
            o, te_p, ms = fit_oof(
                Xtr, ytr, f_in, pd.concat([Xva, X_test], ignore_index=True), kind, params, [seed]
            )
            mu = np.nanmean(ytr)
            F_tr[f"s1_{t}"] = np.where(np.isnan(o), mu, o)
            F_va[f"s1_{t}"] = te_p[: len(Xva)]
            F_te[f"s1_{t}"] = te_p[len(Xva) :]
        es_fold = 0
        for j, t in enumerate(targets):
            use = [c for c in F_tr.columns if include_self or c != f"s1_{t}"]
            A_tr = pd.concat([Xtr, F_tr[use]], axis=1)
            A_va = pd.concat([Xva, F_va[use]], axis=1)
            A_te = pd.concat([X_test.reset_index(drop=True), F_te[use].reset_index(drop=True)], axis=1)
            y = Y[t].values[tr]
            obs = ~np.isnan(y)
            es = (f_in == es_fold) & obs
            fit_m = (f_in != es_fold) & obs
            if es.sum() == 0 or fit_m.sum() < 20:
                continue
            m = fit(A_tr[fit_m], y[fit_m], A_tr[es], y[es], params, seed)
            oof[va, j] = m.predict(A_va)
            test[:, j] += m.predict(A_te) / K
    return pd.DataFrame(oof, columns=targets), pd.DataFrame(test, columns=targets)
