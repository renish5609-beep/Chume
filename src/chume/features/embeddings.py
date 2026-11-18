from __future__ import annotations

import numpy as np

S2_13 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


LOCAL_WEIGHTS = {
    "ResNet50_Weights.SENTINEL2_ALL_MOCO": "data/weights/resnet50_sentinel2_all_moco-df8b932e.pth",
}


def load_encoder(weight_name: str = "ResNet50_Weights.SENTINEL2_ALL_MOCO"):
    import torch

    try:
        from torchgeo.models import get_weight, resnet18, resnet50
    except ImportError:
        # torchgeo not installed: load the same published SSL4EO-S12 checkpoint
        # (downloaded from hf.co/torchgeo) into an identical timm ResNet50.
        import timm

        model = timm.create_model("resnet50", in_chans=13, num_classes=0)
        sd = torch.load(LOCAL_WEIGHTS[weight_name], map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        model.eval()
        return model, 13

    w = get_weight(weight_name)
    arch = w.meta.get("model", "resnet50")
    model = (resnet50 if arch == "resnet50" else resnet18)(weights=w)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, w.meta.get("in_chans", 13)


def to_13(bands: dict[str, np.ndarray]) -> np.ndarray:
    h, w = next(iter(bands.values())).shape
    out = np.zeros((13, h, w), dtype=np.float32)
    for i, b in enumerate(S2_13):
        if b in bands:
            out[i] = bands[b]
        elif b == "B10":
            out[i] = 0.0
        elif b == "B09" and "B8A" in bands:
            out[i] = bands["B8A"]
    return out


def embed_batch(
    model, chips_13: list[np.ndarray], size: int = 224, center_crop: float | None = None
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    xs = []
    for a in chips_13:
        x = torch.from_numpy(a)[None]
        if center_crop:
            h, w = x.shape[-2:]
            ch, cw = int(h * center_crop), int(w * center_crop)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            x = x[..., y0 : y0 + ch, x0 : x0 + cw]
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        xs.append(x)
    with torch.no_grad():
        z = model(torch.cat(xs))
    return z.cpu().numpy()


def main() -> None:
    import argparse

    import pandas as pd
    from sklearn.decomposition import PCA

    from ..chips import Chip
    from ..config import load_config
    from ..io import find_chip, load_chip

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--features", default="data/processed/features_true.parquet")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    cfg = load_config(a.config)
    c, cc, fc = cfg["columns"], cfg["chips"], cfg["features"]
    df = pd.read_parquet(a.features)
    model, _ = load_encoder(fc["embeddings_model"])
    Z_g, Z_c = [], []
    ids = df[c["id"]].astype(str).tolist()
    for i in range(0, len(ids), a.batch):
        batch = []
        for sid in ids[i : i + a.batch]:
            p = find_chip(cfg["paths"]["chip_dir"], sid, cc["pattern"])
            chip = Chip(load_chip(p), cc["band_order"], cc["scale"], cc["scl_bad"], cc.get("cldprb_bad", 40.0))
            batch.append(to_13(chip.bands))
        Z_g.append(embed_batch(model, batch))
        Z_c.append(embed_batch(model, batch, center_crop=0.5))
    Z_g, Z_c = np.concatenate(Z_g), np.concatenate(Z_c)
    is_tr = (df["_split"] == "train").values
    k = fc["embeddings_pca"]
    for name, Z in (("g", Z_g), ("c", Z_c)):
        pca = PCA(n_components=k, random_state=cfg["seed"]).fit(Z[is_tr])
        P = pca.transform(Z)
        for j in range(k):
            df[f"emb_{name}{j}"] = P[:, j]
    df.to_parquet(a.features, index=False)
    print(f"added {2*k} embedding columns to {a.features}")


if __name__ == "__main__":
    main()
