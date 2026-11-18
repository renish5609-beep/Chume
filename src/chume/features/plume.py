from __future__ import annotations

import numpy as np
import pandas as pd

from ..chips import Chip


def wind_uv(row: pd.Series, cols: dict) -> tuple[float, float]:
    if cols.get("wind_u") and cols.get("wind_v"):
        return float(row[cols["wind_u"]]), float(row[cols["wind_v"]])
    if cols.get("wind_speed") and cols.get("wind_dir"):
        ws = float(row[cols["wind_speed"]])
        d = np.deg2rad(float(row[cols["wind_dir"]]))
        conv = cols.get("wind_dir_convention", "meteorological_from")
        if conv == "meteorological_from":
            return -ws * np.sin(d), -ws * np.cos(d)
        if conv == "meteorological_to":
            return ws * np.sin(d), ws * np.cos(d)
        if conv == "math_to":
            return ws * np.cos(d), ws * np.sin(d)
        raise ValueError(conv)
    return float("nan"), float("nan")


def polar_grid(h: int, w: int, north_up: bool = True) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dx = xx - cx
    dy = (cy - yy) if north_up else (yy - cy)
    r = np.sqrt(dx**2 + dy**2) / (min(h, w) / 2.0)
    theta = np.arctan2(dy, dx)
    return r, theta


def region_masks(
    h: int, w: int, wind_angle: float, radii: list[float], half_angle_deg: float, north_up: bool = True
):
    r, theta = polar_grid(h, w, north_up)
    rel = np.angle(np.exp(1j * (theta - wind_angle)))
    a = np.deg2rad(half_angle_deg)
    down = np.abs(rel) <= a
    up = np.abs(rel) >= np.pi - a
    cross = (np.abs(rel) >= np.pi / 2 - a) & (np.abs(rel) <= np.pi / 2 + a)
    out = {}
    r_core = radii[0]
    out["core"] = r <= r_core
    for rad in radii:
        ring = (r > r_core) & (r <= rad) if rad > r_core else (r <= rad)
        out[f"down_{rad}"] = ring & down
        out[f"up_{rad}"] = ring & up
        out[f"cross_{rad}"] = ring & cross
        out[f"ring_{rad}"] = ring
    return out


def _stat(v: np.ndarray) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(v.mean()), float(np.percentile(v, 90)), float(v.max())


def plume_features(
    chip: Chip, u: float, v: float, cfg: dict, rng: np.random.RandomState | None = None
) -> dict[str, float]:
    fc = cfg["features"]
    radii = fc["plume_radii"]
    ws = float(np.hypot(u, v)) if np.isfinite(u) and np.isfinite(v) else np.nan
    angle = float(np.arctan2(v, u)) if np.isfinite(ws) else 0.0
    if fc.get("plume_random_angle"):
        rng = rng or np.random.RandomState(0)
        angle = float(rng.uniform(-np.pi, np.pi))
    calm = (not np.isfinite(ws)) or ws < fc["calm_ws"]
    masks = region_masks(
        chip.h, chip.w, angle, radii, fc["plume_half_angle_deg"], cfg["chips"].get("north_up", True)
    )
    clear = chip.mask
    feats: dict[str, float] = {"is_calm": float(calm)}
    layers = chip.layers()
    bg_std = {}
    for name, L in layers.items():
        bg = L[clear & ~masks["core"]]
        bg = bg[np.isfinite(bg)]
        bg_std[name] = float(bg.std()) if bg.size > 10 else np.nan
    for name, L in layers.items():
        core_m, core_p90, core_max = _stat(L[masks["core"] & clear])
        feats[f"p_{name}_core_mean"], feats[f"p_{name}_core_p90"], feats[f"p_{name}_core_max"] = (
            core_m,
            core_p90,
            core_max,
        )
        for rad in radii:
            st = {}
            for reg in ("down", "up", "cross", "ring"):
                st[reg] = _stat(L[masks[f"{reg}_{rad}"] & clear])
            for reg in ("down", "up", "cross"):
                feats[f"p_{name}_{reg}{rad}_mean"] = st[reg][0]
                feats[f"p_{name}_{reg}{rad}_p90"] = st[reg][1]
            d_minus_u = st["down"][0] - st["up"][0]
            feats[f"p_{name}_du{rad}"] = d_minus_u
            feats[f"p_{name}_dc{rad}"] = st["down"][0] - st["cross"][0]
            feats[f"p_{name}_cr{rad}"] = core_m - st["ring"][0]
            s = bg_std[name]
            feats[f"p_{name}_duz{rad}"] = d_minus_u / s if s and np.isfinite(s) and s > 0 else np.nan
            if calm:
                feats[f"p_{name}_du{rad}"] = np.nan
                feats[f"p_{name}_duz{rad}"] = np.nan
    if "hot_ratio" in layers:
        hr = layers["hot_ratio"]
        for thr in (1.2, 1.5, 2.0):
            feats[f"hot_n{thr}_core"] = float(((hr > thr) & masks["core"]).sum())
            feats[f"hot_n{thr}_all"] = float((hr > thr).sum())
        feats["hot_ratio_max"] = float(np.nanmax(hr))
    if "tai" in layers:
        tai = layers["tai"]
        feats["tai_n_pos_all"] = float((tai > 0.45).sum())
        feats["tai_max"] = float(np.nanmax(tai))
    for rad in radii:
        feats[f"clear_down{rad}"] = (
            float(clear[masks[f"down_{rad}"]].mean()) if masks[f"down_{rad}"].any() else np.nan
        )
        feats[f"clear_up{rad}"] = (
            float(clear[masks[f"up_{rad}"]].mean()) if masks[f"up_{rad}"].any() else np.nan
        )
    feats["clear_core"] = float(clear[masks["core"]].mean()) if masks["core"].any() else np.nan
    return feats
