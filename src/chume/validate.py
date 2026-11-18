from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def validate_submission(sub: pd.DataFrame, test_ids: pd.Series, id_col: str, targets: list[str]) -> None:
    exp = [id_col] + targets
    assert list(sub.columns) == exp, f"columns {list(sub.columns)} != {exp}"
    assert len(sub) == len(test_ids), f"rows {len(sub)} != {len(test_ids)}"
    assert sub[id_col].map(type).eq(str).all(), "sample_id must be str"
    assert not sub[id_col].duplicated().any(), "duplicate ids"
    assert set(sub[id_col]) == set(test_ids.astype(str)), "id mismatch with test"
    vals = sub[targets].to_numpy(dtype=float)
    assert np.isfinite(vals).all(), "non-finite predictions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--id", default="sample_id")
    a = ap.parse_args()
    targets = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
    sub = pd.read_csv(a.submission, dtype={a.id: str})
    test = pd.read_csv(a.test, dtype={a.id: str})
    validate_submission(sub, test[a.id], a.id, targets)
    print("submission OK")


if __name__ == "__main__":
    main()
