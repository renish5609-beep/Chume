from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def availability_code(df: pd.DataFrame, targets: list[str]) -> np.ndarray:
    bits = np.stack([df[t].notna().values.astype(int) for t in targets], axis=1)
    return (bits * (2 ** np.arange(len(targets)))).sum(axis=1)


def make_folds(
    df: pd.DataFrame, site_col: str, targets: list[str], n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    groups = df[site_col].astype(str).values
    strat = availability_code(df, targets)
    counts = pd.Series(strat).value_counts()
    strat = np.where(pd.Series(strat).map(counts).values < n_folds, -1, strat)
    folds = np.full(len(df), -1, dtype=int)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = splitter.split(df, strat, groups)
    except ValueError:
        rng = np.random.RandomState(seed)
        uniq = np.unique(groups)
        perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
        splits = GroupKFold(n_splits=n_folds).split(df, groups=[perm[g] for g in groups])
    for k, (_, va) in enumerate(splits):
        folds[va] = k
    assert (folds >= 0).all()
    return folds


def check_group_integrity(folds: np.ndarray, groups: np.ndarray) -> None:
    g = pd.DataFrame({"f": folds, "g": groups})
    n = g.groupby("g")["f"].nunique()
    bad = n[n > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} sites appear in more than one fold")
