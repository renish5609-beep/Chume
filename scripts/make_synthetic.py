import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"]
T = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--size", type=int, default=48)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    out = Path(a.out)
    (out / "chips").mkdir(parents=True, exist_ok=True)
    n = a.n_train + a.n_test
    n_sites = n // 8
    site = rng.randint(0, n_sites, n)
    split = np.where(site < int(n_sites * 0.77), "train", "test")
    emis = rng.gamma(1.5, 1.0, n_sites)[site] * (0.5 + rng.rand(n))
    flare = (rng.rand(n_sites) < 0.4)[site] * rng.gamma(2.0, 1.0, n)
    ws = rng.gamma(2.5, 1.5, n)
    wd = rng.uniform(0, 360, n)
    h = w = a.size
    yy, xx = np.mgrid[0:h, 0:w]
    rows = []
    for i in range(n):
        sid = f"S{i:05d}"
        arr = (rng.rand(len(BANDS), h, w) * 0.05 + 0.2).astype(np.float32)
        th = np.deg2rad(wd[i])
        ux, vy = -np.sin(th), -np.cos(th)
        dx, dy = xx - (w - 1) / 2, ((h - 1) / 2) - yy
        along = dx * ux + dy * vy
        across = np.abs(-dx * vy + dy * ux)
        pl = (along > 0) & (across < 4 + along * 0.2)
        conc = emis[i] / max(ws[i], 0.5)
        arr[BANDS.index("B12")][pl] *= np.exp(-0.03 * conc)
        if flare[i] > 0:
            c = (np.abs(xx - w // 2) <= 1) & (np.abs(yy - h // 2) <= 1)
            arr[BANDS.index("B12")][c] += 0.3 * flare[i]
            arr[BANDS.index("B11")][c] += 0.1 * flare[i]
        scl = np.full((h, w), 4.0, dtype=np.float32)
        cl = rng.rand() * 0.4
        scl[rng.rand(h, w) < cl] = 8
        arr[BANDS.index("SCL")] = scl
        arr[:-1] *= 10000
        np.save(out / "chips" / f"{sid}.npy", arr)
        r = {
            "sample_id": sid,
            "site_key": f"site_{site[i]:04d}",
            "wind_speed": ws[i],
            "wind_dir": wd[i],
            "temp": rng.randn(),
            "elev": rng.randn() + site[i] % 7,
            "_split": split[i],
        }
        for k in range(22):
            r[f"aux_{k}"] = rng.randn()
        r["t_ch4"] = conc * 10 + rng.randn() * 3
        r["t_no2"] = 0.5 * flare[i] + 0.2 * conc + rng.randn() * 0.5
        r["t_co"] = 0.3 * flare[i] + 0.1 * conc + rng.randn() * 0.5
        r["t_aai"] = 0.1 * flare[i] + rng.randn() * 0.6
        r["t_firms"] = 5 * flare[i] + rng.randn()
        for t in T:
            if rng.rand() < cl + 0.1:
                r[t] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    tr = df[df._split == "train"].drop(columns="_split")
    te = df[df._split == "test"].drop(columns=["_split"] + T)
    tr.to_csv(out / "train.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    print(tr.shape, te.shape)


if __name__ == "__main__":
    main()
