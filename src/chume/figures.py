from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features.plume import region_masks


def ablation_bar(abl_json: str, out: Path) -> None:
    rows = json.loads(Path(abl_json).read_text())
    names = [r[0] for r in rows]
    m = [r[1] for r in rows]
    s = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(names, m, xerr=s, color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlabel("MCRMSE (site-grouped CV, lower is better)")
    lo = min(m) - 3 * max(s + [1e-6])
    ax.set_xlim(left=max(0, lo))
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=300)


def region_diagram(out: Path) -> None:
    h = w = 128
    ms = region_masks(h, w, np.deg2rad(35), [0.10, 0.25, 0.45], 30)
    img = np.zeros((h, w, 3))
    img[ms["down_0.45"]] = [0.85, 0.3, 0.2]
    img[ms["up_0.45"]] = [0.2, 0.45, 0.85]
    img[ms["cross_0.45"]] = [0.55, 0.55, 0.55]
    img[ms["core"]] = [1.0, 0.85, 0.1]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(img)
    ax.annotate(
        "",
        xy=(64 + 40 * np.cos(np.deg2rad(35)), 64 - 40 * np.sin(np.deg2rad(35))),
        xytext=(64, 64),
        arrowprops=dict(arrowstyle="->", color="k", lw=2),
    )
    ax.set_title("Core (yellow), downwind (red),\nupwind (blue), crosswind (grey)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "regions.png", dpi=300)


def oof_scatter(oof_csv: str, train_csv: str, id_col: str, targets: list[str], out: Path) -> None:
    o = pd.read_csv(oof_csv, dtype={id_col: str})
    t = pd.read_csv(train_csv, dtype={id_col: str})[[id_col] + targets]
    d = o.merge(t, on=id_col, suffixes=("_pred", ""))
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4))
    for ax, tg in zip(axes, targets):
        m = d[tg].notna()
        ax.scatter(d.loc[m, tg], d.loc[m, f"{tg}_pred"], s=6, alpha=0.5)
        lo, hi = np.nanpercentile(d.loc[m, tg], [1, 99])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(tg)
        ax.set_xlabel("actual")
        ax.set_ylabel("OOF predicted")
    fig.tight_layout()
    fig.savefig(out / "oof_scatter.png", dpi=300)


def target_panels(train_csv: str, targets: list[str], out: Path) -> None:
    t = pd.read_csv(train_csv)[targets]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    c = t.corr()
    im = axes[0].imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(targets)), targets, rotation=45)
    axes[0].set_yticks(range(len(targets)), targets)
    for i in range(len(targets)):
        for j in range(len(targets)):
            axes[0].text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title("Target correlation (co-observed)")
    axes[1].bar(targets, t.isna().mean() * 100, color="#E45756")
    axes[1].set_ylabel("% missing")
    axes[1].set_title("Missing targets")
    fig.tight_layout()
    fig.savefig(out / "targets.png", dpi=300)


def importance_plot(imp_csv: str, out: Path, k: int = 20) -> None:
    imp = pd.read_csv(imp_csv, index_col=0)
    n = imp.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    for ax, col in zip(np.atleast_1d(axes), imp.columns):
        s = imp[col].sort_values(ascending=False).head(k)[::-1]
        ax.barh(s.index, s.values, color="#54A24B")
        ax.set_title(col)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "importance.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train", default="data/raw/train.csv")
    ap.add_argument("--ablations", default="docs/ABLATIONS.json")
    ap.add_argument("--id", default="sample_id")
    ap.add_argument("--out", default="outputs/figures")
    a = ap.parse_args()
    targets = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    region_diagram(out)
    target_panels(a.train, targets, out)
    oof_scatter(str(Path(a.run) / "oof.csv"), a.train, a.id, targets, out)
    importance_plot(str(Path(a.run) / "importance_lgbm.csv"), out)
    if Path(a.ablations).exists():
        ablation_bar(a.ablations, out)
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
