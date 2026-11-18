from __future__ import annotations

import pandas as pd


def add_site_context(
    df: pd.DataFrame, site_col: str, cols: list[str], split_col: str = "_split"
) -> pd.DataFrame:
    df = df.copy()
    keys = [split_col, site_col]
    means = df.groupby(keys)[cols].transform("mean")
    counts = df.groupby(keys)[cols[0]].transform("size")
    for c in cols:
        df[f"sc_{c}_mean"] = means[c]
        df[f"sc_{c}_dev"] = df[c] - means[c]
    df["sc_n_rows"] = counts
    return df
