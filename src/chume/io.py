from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str | Path, id_col: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={id_col: str})


def find_chip(chip_dir: str | Path, sample_id: str, pattern: str = "{id}.*") -> str | None:
    hits = glob.glob(os.path.join(str(chip_dir), "**", pattern.format(id=sample_id)), recursive=True)
    return sorted(hits)[0] if hits else None


def load_chip(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        arr = z[list(z.keys())[0]]
    elif ext in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                arr = src.read()
        except ImportError:
            import tifffile

            arr = tifffile.imread(path)
    elif ext in (".png", ".jpg", ".jpeg"):
        from PIL import Image

        arr = np.asarray(Image.open(path))
    else:
        raise ValueError(f"unknown chip format {ext}")
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim == 3 and arr.shape[-1] < arr.shape[0] and arr.shape[-1] <= 20:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def run_dir(root: str | Path, run_id: str) -> Path:
    p = Path(root) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | Path, rec: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
