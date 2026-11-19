from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, seed_everything
from .cv import check_group_integrity, make_folds
from .features.massbalance import target_forward, target_inverse
from .io import append_jsonl, run_dir
from .metric import mcrmse, rmse_masked
from .models.blend import apply_blend, fit_blend, nested_blend_oof
from .models.chain import strict_chain
from .models.gate import apply_gate, fit_gate, nested_gate_oof
from .models.gbdt import fit_oof, importance

NON_FEATURE_PREFIX = ("_",)


def feature_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    import re

    c = cfg["columns"]
    skip = {c["id"], c["site"], *c["targets"], *c.get("drop", [])}
    drop_re = [re.compile(r) for r in cfg["features"].get("drop_regex", []) or []]
    keep_re = [re.compile(r) for r in cfg["features"].get("keep_regex", []) or []]
    cols = []
    for col in df.columns:
        if col in skip or col.startswith(NON_FEATURE_PREFIX):
            continue
        if any(r.search(col) for r in drop_re) and not any(r.search(col) for r in keep_re):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not cfg["features"].get("site_context") and col.startswith("sc_"):
            continue
        if not cfg["features"].get("plume", True) and (
            col.startswith(("p_", "hot_", "tai_", "clear_", "is_calm")) or col.startswith("mb_p_")
        ):
            continue
        if (
            not cfg["features"].get("massbalance", True)
            and col.startswith("mb_")
            and col not in ("mb_u", "mb_v", "mb_ws")
        ):
            continue
        if not cfg["features"].get("generic_chip", True) and (
            col.startswith(("g_", "scl_")) or col == "clear_frac"
        ):
            continue
        if not cfg["features"].get("embeddings", False) and col.startswith("emb_"):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        cols.append(col)
    return cols


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


def pick_transform(X, y, ws, folds, Xte, cfg, seed) -> str:
    mode = cfg["targets"]["transform"]
    if mode != "auto":
        return mode
    best, best_s = "none", np.inf
    obs = ~np.isnan(y)
    for kind in cfg["targets"]["candidates"]:
        if kind == "slog" and np.nanmin(np.abs(y)) == np.inf:
            continue
        z = np.where(obs, target_forward(np.nan_to_num(y), kind, ws), np.nan)
        o, _, _ = fit_oof(X, z, folds, Xte.iloc[:1], "lgbm", cfg["lgbm_params"], [seed])
        p = target_inverse(o, kind, ws)
        s = rmse_masked(y, p)
        if s < best_s:
            best, best_s = kind, s
    return best


def run(cfg: dict, features_path: str, run_id: str) -> dict:
    t0 = time.time()
    seed = cfg["seed"]
    seed_everything(seed)
    c = cfg["columns"]
    targets = c["targets"]
    df = pd.read_parquet(features_path)
    df[c["id"]] = df[c["id"]].astype(str)
    tr = df[df["_split"] == "train"].reset_index(drop=True)
    te = df[df["_split"] == "test"].reset_index(drop=True)
    cols = feature_columns(df, cfg)
    X, Xte = tr[cols], te[cols]
    Y = tr[targets]
    folds = make_folds(tr, c["site"], targets, cfg["cv"]["n_folds"], seed)
    check_group_integrity(folds, tr[c["site"]].astype(str).values)
    out = run_dir(cfg["paths"]["runs"], run_id)
    pd.DataFrame({c["id"]: tr[c["id"]], "fold": folds}).to_csv(out / "folds.csv", index=False)
    ws_tr = tr["mb_ws"].fillna(tr["mb_ws"].median()).values
    ws_te = te["mb_ws"].fillna(tr["mb_ws"].median()).values
    mseeds = [seed + i for i in range(cfg["models"]["n_model_seeds"])]

    stages: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    transforms, imps = {}, {}
    for t in targets:
        y = Y[t].values.astype(float)
        kind = pick_transform(X, y, ws_tr, folds, Xte, cfg, seed)
        transforms[t] = kind
        obs = ~np.isnan(y)
        z = np.where(obs, target_forward(np.nan_to_num(y), kind, ws_tr), np.nan)
        for mk in ("lgbm", "catboost"):
            if not cfg["models"].get(mk, True):
                continue
            params = cfg["lgbm_params"] if mk == "lgbm" else cfg["catboost_params"]
            o, p, ms = fit_oof(X, z, folds, Xte, mk, params, mseeds)
            stages.setdefault(mk, {})[t] = (
                target_inverse(o, kind, ws_tr),
                target_inverse(p, kind, ws_te),
            )
            if mk == "lgbm":
                imps[t] = importance(ms, cols)

    if cfg["models"].get("chain", True):
        co, ct = strict_chain(
            X, Y, folds, Xte, cfg["lgbm_params"], seed, cfg["models"].get("chain_include_self", False)
        )
        stages["chain"] = {t: (co[t].values, ct[t].values) for t in targets}

    final_oof = pd.DataFrame(index=tr.index)
    final_test = pd.DataFrame(index=te.index)
    report: dict = {"stages": {}, "blend_w": {}, "gate_w": {}, "transforms": transforms}
    for name, d in stages.items():
        o = pd.DataFrame({t: d[t][0] for t in targets})
        report["stages"][name] = mcrmse(Y, o, targets)
    mean_o = pd.DataFrame({t: np.full(len(tr), np.nanmean(Y[t])) for t in targets})
    report["stages"]["mean"] = mcrmse(Y, mean_o, targets)

    names = list(stages)
    for t in targets:
        y = Y[t].values.astype(float)
        P = np.column_stack([stages[n][t][0] for n in names])
        Pt = np.column_stack([stages[n][t][1] for n in names])
        P = np.where(np.isnan(P), np.nanmean(y), P)
        if cfg["models"].get("blend", True) and len(names) > 1:
            bo, _ = nested_blend_oof(y, P, folds)
            w = fit_blend(y, P)
            bt = apply_blend(Pt, w)
        else:
            j0 = names.index("chain") if "chain" in names else 0
            bo, bt, w = P[:, j0], Pt[:, j0], np.eye(len(names))[j0]
        report["blend_w"][t] = dict(zip(names, map(float, w)))
        if cfg["models"].get("gate", True):
            go, gws = nested_gate_oof(y, bo, folds)
            mu = float(np.nanmean(y))
            gw = fit_gate(y, bo, mu)
            gt = apply_gate(bt, mu, gw)
            report["gate_w"][t] = {"full": gw, "per_fold": gws}
        else:
            go, gt = bo, bt
        final_oof[t], final_test[t] = go, gt

    if cfg.get("post", {}).get("clip_train_range", False):
        for t in targets:
            lo, hi = float(np.nanmin(Y[t])), float(np.nanmax(Y[t]))
            final_oof[t] = np.clip(final_oof[t].values, lo, hi)
            final_test[t] = np.clip(final_test[t].values, lo, hi)
    report["final"] = mcrmse(Y, final_oof, targets)
    sub = pd.DataFrame({c["id"]: te[c["id"]].astype(str)})
    for t in targets:
        v = final_test[t].values
        sub[t] = np.where(np.isfinite(v), v, np.nanmean(Y[t]))
    sub.to_csv(out / "submission.csv", index=False)
    oof_df = final_oof.copy()
    oof_df.insert(0, c["id"], tr[c["id"]])
    oof_df.to_csv(out / "oof.csv", index=False)
    pd.DataFrame(imps).to_csv(out / "importance_lgbm.csv")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    report.update(
        {
            "run_id": run_id,
            "features": features_path,
            "n_features": len(cols),
            "git": git_hash(),
            "seconds": time.time() - t0,
        }
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    append_jsonl(
        Path(cfg["paths"]["runs"]).parent / "experiments.jsonl",
        {
            "run_id": run_id,
            "final": report["final"][0],
            "per_target": report["final"][1],
            "stages": {k: v[0] for k, v in report["stages"].items()},
            "seconds": report["seconds"],
        },
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--features", default="data/processed/features_true.parquet")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    over = {"seed": a.seed} if a.seed is not None else None
    cfg = load_config(a.config, over)
    rid = a.run_id or time.strftime("%Y%m%d_%H%M%S")
    rep = run(cfg, a.features, rid)
    print(
        json.dumps(
            {"final": rep["final"], "stages": {k: v[0] for k, v in rep["stages"].items()}},
            indent=2,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
