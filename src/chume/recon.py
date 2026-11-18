from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .io import find_chip, load_chip, read_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="docs/DATA_NOTES_auto.md")
    ap.add_argument("--n_chips", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config)
    c = cfg["columns"]
    tr = read_table(cfg["paths"]["train_csv"], c["id"])
    te = read_table(cfg["paths"]["test_csv"], c["id"])
    L = ["# Auto recon", ""]
    raw = Path(cfg["paths"]["raw"])
    L += ["## Files", ""] + [
        f"- {p.relative_to(raw)}  ({p.stat().st_size/1e6:.1f} MB)"
        for p in sorted(raw.glob("*"))
        if p.is_file()
    ]
    L += [
        "",
        f"train {tr.shape}, test {te.shape}",
        "",
        "## Columns",
        "",
        "| col | dtype | train_na% | test_na% | train min/mean/max | test mean |",
        "|---|---|---|---|---|---|",
    ]
    for col in tr.columns:
        tn = tr[col].isna().mean() * 100
        ten = te[col].isna().mean() * 100 if col in te else float("nan")
        if pd.api.types.is_numeric_dtype(tr[col]):
            s = f"{tr[col].min():.4g}/{tr[col].mean():.4g}/{tr[col].max():.4g}"
            tm = f"{te[col].mean():.4g}" if col in te and pd.api.types.is_numeric_dtype(te[col]) else "-"
        else:
            s, tm = f"{tr[col].nunique()} unique", "-"
        L.append(f"| {col} | {tr[col].dtype} | {tn:.1f} | {ten:.1f} | {s} | {tm} |")
    T = [t for t in c["targets"] if t in tr]
    if T:
        L += [
            "",
            "## Targets",
            "",
            tr[T].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_markdown(),
            "",
        ]
        L += ["Skew:", "", tr[T].skew().to_frame("skew").to_markdown(), ""]
        L += ["Pairwise corr (co-observed rows):", "", tr[T].corr().round(3).to_markdown(), ""]
        pat = tr[T].notna().astype(int).astype(str).agg("".join, axis=1).value_counts()
        L += [
            "Availability patterns (order " + ",".join(T) + "):",
            "",
            pat.to_frame("n").to_markdown(),
            "",
        ]
    if c["site"] in tr:
        L += [
            f"Sites: train {tr[c['site']].nunique()}, "
            f"test {te[c['site']].nunique() if c['site'] in te else 'n/a'}",
            f"Overlap: {len(set(tr[c['site']]) & set(te.get(c['site'], pd.Series(dtype=str))))}",
            f"Rows per site (train): {tr[c['site']].value_counts().describe().round(2).to_dict()}",
            "",
        ]
    L += ["## Chips", ""]
    for sid in tr[c["id"]].head(5):
        p = find_chip(cfg["paths"]["chip_dir"], sid, cfg["chips"]["pattern"])
        if p is None:
            L.append(f"- {sid}: NOT FOUND")
            continue
        arr = load_chip(p)
        L.append(
            f"- {sid}: {p} shape={arr.shape} dtype={arr.dtype} "
            f"per-band max={np.nanmax(arr.reshape(arr.shape[0], -1), axis=1).round(1).tolist()}"
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = a.n_chips
        fig, axes = plt.subplots(4, n // 4 + (n % 4 > 0), figsize=(16, 13))
        bo = cfg["chips"]["band_order"]
        for ax, sid in zip(axes.ravel(), tr[c["id"]].sample(n, random_state=0)):
            p = find_chip(cfg["paths"]["chip_dir"], sid, cfg["chips"]["pattern"])
            ax.axis("off")
            if p is None:
                continue
            arr = load_chip(p).astype(float)
            idx = [bo.index(b) for b in ("B12", "B11", "B04") if b in bo and bo.index(b) < arr.shape[0]]
            if len(idx) == 3:
                rgb = np.stack([arr[i] for i in idx], -1)
                rgb = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)
                ax.imshow(rgb)
            ax.plot(arr.shape[2] / 2, arr.shape[1] / 2, "c+", ms=10)
        Path("outputs/recon").mkdir(parents=True, exist_ok=True)
        fig.savefig("outputs/recon/contact_sheet_swir.png", dpi=110, bbox_inches="tight")
        L += [
            "",
            "Contact sheet (B12/B11/B04 false color, cyan + at center): "
            "outputs/recon/contact_sheet_swir.png",
        ]
    except Exception as e:
        L.append(f"contact sheet failed: {e}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
