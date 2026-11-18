from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import load_config
from .train import feature_columns


def adversarial(df: pd.DataFrame, cols: list[str], seed: int = 42) -> tuple[float, pd.Series]:
    y = (df["_split"] == "test").astype(int).values
    X = df[cols]
    oof = np.zeros(len(df))
    imp = np.zeros(len(cols))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=seed
        )
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.booster_.feature_importance("gain")
    return float(roc_auc_score(y, oof)), pd.Series(imp, index=cols).sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--features", default="data/processed/features_true.parquet")
    ap.add_argument("--out", default="outputs/adversarial.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = pd.read_parquet(a.features)
    cols = feature_columns(df, cfg)
    auc, imp = adversarial(df, cols, cfg["seed"])
    res = {"auc": auc, "top_drift": imp.head(25).to_dict()}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
