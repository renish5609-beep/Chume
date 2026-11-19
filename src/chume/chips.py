from __future__ import annotations

import numpy as np

EPS = 1e-6


class Chip:
    def __init__(
        self,
        arr: np.ndarray,
        band_order: list[str],
        scale: float = 10000.0,
        scl_bad=(0, 1, 3, 8, 9, 10),
        cldprb_bad: float = 40.0,
    ):
        arr = arr.astype(np.float32)
        self.bands: dict[str, np.ndarray] = {}
        for i, name in enumerate(band_order[: arr.shape[0]]):
            self.bands[name] = arr[i]
        self.scl = self.bands.pop("SCL", None)
        self.cldprb = self.bands.pop("CLDPRB", None)
        self.cldprb_bad = cldprb_bad
        for k in list(self.bands):
            b = self.bands[k]
            if scale and np.nanmax(b) > 2.0:
                b = b / scale
            self.bands[k] = b
        self.h, self.w = arr.shape[1], arr.shape[2]
        self.mask = self._mask(scl_bad)

    def _mask(self, scl_bad) -> np.ndarray:
        m = np.ones((self.h, self.w), dtype=bool)
        for b in self.bands.values():
            m &= np.isfinite(b)
        if self.scl is not None:
            m &= ~np.isin(self.scl.astype(int), list(scl_bad))
            if self.cldprb is not None:
                m &= self.cldprb < self.cldprb_bad
        else:
            if "B02" in self.bands and "B04" in self.bands:
                bright = (self.bands["B02"] + self.bands["B04"]) / 2
                m &= bright < 0.35
            if "B10" in self.bands:
                m &= self.bands["B10"] < 0.012
        return m

    def get(self, name: str) -> np.ndarray | None:
        return self.bands.get(name)

    def layers(self) -> dict[str, np.ndarray]:
        g = self.get
        out: dict[str, np.ndarray] = {}
        b11, b12, b8a, b08, b04, b01 = g("B11"), g("B12"), g("B8A"), g("B08"), g("B04"), g("B01")
        if b08 is None and b8a is not None:
            b08 = b8a
        if b11 is not None and b12 is not None:
            r = b12 / (b11 + EPS)
            out["r_ch4"] = r
            med = np.nanmedian(r[self.mask]) if self.mask.any() else np.nanmedian(r)
            out["r_ch4_norm"] = r / (med + EPS)
        if b12 is not None and b11 is not None and b8a is not None:
            out["tai"] = (b12 - b11) / (b8a + EPS)
        if b12 is not None and b8a is not None:
            out["hot"] = b12 - b8a
            out["hot_ratio"] = b12 / (b8a + EPS)
        if b08 is not None and b04 is not None:
            out["ndvi"] = (b08 - b04) / (b08 + b04 + EPS)
        if b11 is not None and b08 is not None:
            out["ndbi"] = (b11 - b08) / (b11 + b08 + EPS)
        if b01 is not None and b04 is not None:
            out["haze"] = b01 / (b04 + EPS)
        if b01 is not None:
            out["b01"] = b01
        if b12 is not None:
            out["b12"] = b12
        return out


def generic_stats(chip: Chip) -> dict[str, float]:
    feats: dict[str, float] = {}
    m = chip.mask
    feats["clear_frac"] = float(m.mean())
    for name, b in chip.bands.items():
        v = b[m] if m.any() else b.ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        feats[f"g_{name}_p10"], feats[f"g_{name}_p50"], feats[f"g_{name}_p90"] = (
            float(p10),
            float(p50),
            float(p90),
        )
    if chip.scl is not None:
        scl = chip.scl.astype(int)
        for c in range(12):
            feats[f"scl_{c}"] = float((scl == c).mean())
    return feats
