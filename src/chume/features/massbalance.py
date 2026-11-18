from __future__ import annotations

import numpy as np
import pandas as pd


def add_wind(df: pd.DataFrame, u: np.ndarray, v: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    ws = np.hypot(u, v)
    df["mb_u"], df["mb_v"], df["mb_ws"] = u, v, ws
    df["mb_inv_ws"] = 1.0 / np.clip(ws, 0.5, None)
    df["mb_log_ws"] = np.log(np.clip(ws, 0.1, None))
    return df


def add_massbalance(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    df = df.copy()
    inv = df["mb_inv_ws"]
    ws = df["mb_ws"]
    cand = [
        c
        for c in df.columns
        if c.startswith("hot_n") or c in ("hot_ratio_max", "tai_max", "tai_n_pos_all")
    ]
    cand += [
        c
        for c in df.columns
        if c.startswith("p_tai_core") or c.startswith("p_hot_core") or c.startswith("p_ndbi_core_mean")
    ]
    for c in cand:
        df[f"mb_{c}_per_ws"] = df[c] * inv
    for rad in radii:
        for layer in ("r_ch4_norm", "r_ch4", "haze", "b01"):
            c = f"p_{layer}_du{rad}"
            if c in df.columns:
                df[f"mb_{c}_x_ws"] = df[c] * ws
    return df


def target_forward(y: np.ndarray, kind: str, ws: np.ndarray) -> np.ndarray:
    if kind == "none":
        return y
    if kind == "ws_mult":
        return y * np.clip(ws, 0.5, None)
    if kind == "slog":
        return np.sign(y) * np.log1p(np.abs(y))
    raise ValueError(kind)


def target_inverse(z: np.ndarray, kind: str, ws: np.ndarray) -> np.ndarray:
    if kind == "none":
        return z
    if kind == "ws_mult":
        return z / np.clip(ws, 0.5, None)
    if kind == "slog":
        return np.sign(z) * np.expm1(np.abs(z))
    raise ValueError(kind)
