import numpy as np
import pandas as pd

from chume.chips import Chip
from chume.cv import check_group_integrity, make_folds
from chume.features.plume import plume_features, region_masks, wind_uv
from chume.metric import mcrmse
from chume.models.gate import apply_gate, fit_gate
from chume.validate import validate_submission

T = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]


def test_mcrmse_masked():
    yt = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.0, 2.0, np.nan]})
    yp = pd.DataFrame({"a": [2.0, 100.0, 3.0], "b": [1.0, 2.0, 50.0]})
    s, per = mcrmse(yt, yp, ["a", "b"])
    assert np.isclose(per["a"], np.sqrt(0.5))
    assert np.isclose(per["b"], np.sqrt(0.5))
    assert np.isclose(s, np.sqrt(0.5))


def test_folds_grouped():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"site": rng.randint(0, 60, 600).astype(str)})
    for t in T:
        df[t] = np.where(rng.rand(600) < 0.3, np.nan, rng.randn(600))
    f = make_folds(df, "site", T, 5, 0)
    check_group_integrity(f, df["site"].values)
    assert set(f) == set(range(5))


def test_downwind_region_points_with_wind():
    ms = region_masks(65, 65, 0.0, [0.1, 0.25, 0.45], 30)
    ys, xs = np.where(ms["down_0.45"])
    assert xs.mean() > 32
    ms = region_masks(65, 65, np.pi / 2, [0.1, 0.25, 0.45], 30, north_up=True)
    ys, xs = np.where(ms["down_0.45"])
    assert ys.mean() < 32


def test_meteorological_wind():
    u, v = wind_uv(
        pd.Series({"ws": 5.0, "wd": 270.0}),
        {"wind_speed": "ws", "wind_dir": "wd", "wind_dir_convention": "meteorological_from"},
    )
    assert u > 4.9 and abs(v) < 1e-6


def test_plume_contrast_detects_synthetic_plume():
    h = w = 64
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    arr = np.full((len(bands), h, w), 0.2, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    plume = (xx > 36) & (np.abs(yy - 31.5) < 6)
    arr[bands.index("B12")][plume] *= 0.9
    cfg = {
        "features": {
            "plume_radii": [0.1, 0.25, 0.45],
            "plume_half_angle_deg": 30,
            "calm_ws": 1.0,
            "plume_random_angle": False,
        },
        "chips": {"north_up": True},
    }
    chip = Chip(arr, bands, scale=10000.0)
    f_true = plume_features(chip, 5.0, 0.0, cfg)
    f_wrong = plume_features(chip, -5.0, 0.0, cfg)
    assert f_true["p_r_ch4_du0.45"] < 0
    assert f_wrong["p_r_ch4_du0.45"] > 0


def test_gate_shrinks_noise():
    rng = np.random.RandomState(0)
    y = rng.randn(500)
    p = rng.randn(500)
    w = fit_gate(y, p, 0.0)
    assert w < 0.2
    assert np.allclose(apply_gate(p, 0.0, 0.0), 0.0)


def test_validator():
    ids = pd.Series(["001", "002"])
    sub = pd.DataFrame({"sample_id": ["001", "002"], **{t: [0.1, 0.2] for t in T}})
    validate_submission(sub, ids, "sample_id", T)
    sub.loc[0, "t_co"] = np.nan
    try:
        validate_submission(sub, ids, "sample_id", T)
        raise AssertionError("should fail")
    except AssertionError as e:
        assert "non-finite" in str(e)
