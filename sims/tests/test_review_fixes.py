import warnings

import numpy as np
import pytest

from capsim import intake, measure as M, stats as S, sweeps as W, propagate as P, calibration as C
from capsim.scope import ScopeParams


def test_s5_gate_can_fail():
    ba = dict(bias=2.42, sd=3.0, n=40)
    assert S.inside_limits(2.5, ba) and not S.inside_limits(6.0, ba) and not S.inside_limits(-3.0, ba)


def test_unresolved_limbs_give_nan_not_span():
    sc = ScopeParams(um_per_px=1.0, sigma_um=8.0)
    img, px, cx, roi, _ = W.single_loop_image(sc, dict(limb_um=12, apical_um=12, total_um=40, limb_len=100.0), np.random.default_rng(0))
    w = M.limb_widths_fwhm(img, px, roi)
    assert np.all(np.isnan(w)) or np.all(w < 30)         # never the 49 um two-limb span
    out = P.monte_carlo(12, [np.nan, np.nan])
    assert np.isnan(out["resistance"]["mean"])


def test_apical_band_centred_on_column():
    sc = ScopeParams(um_per_px=4.0, sigma_um=2.0, read_noise=0.001, photons=1e5)
    img, px, cx, _, _ = W.single_loop_image(sc, dict(limb_um=12, apical_um=30, total_um=40, limb_len=100.0), np.random.default_rng(0))
    assert M.apical_width_fwhm(img, px, cx) == pytest.approx(30, abs=2.0)


def test_edge_apex_is_counted():
    sc = ScopeParams(um_per_px=1.0, sigma_um=1.5, read_noise=0.002, photons=1e5, illum_strength=0.0)
    rng = np.random.default_rng(0)
    img, px, n = W.row_image(sc, 7, rng, 1.0)
    full = M.count_apices(img, px)
    assert full >= 6
    assert M.count_apices(img[:, 5:], px) >= full - 1          # trimming a few px at the edge must not lose a loop


def test_max_sigma_requires_all_smaller_to_pass():
    import pandas as pd
    df = pd.DataFrame(dict(sigma=[1, 2, 3, 4], err_rate=[0.0, 0.2, 0.0, 0.0]))
    assert W.max_sigma_for_error(df)["max_sigma_um"] == 1.0
    ok = pd.DataFrame(dict(sigma=[1, 2], err_rate=[0, 0]))
    assert W.max_sigma_for_error(ok) == dict(max_sigma_um=2.0, limit_reached=False)


def test_roc_uses_adjacent_densities_and_ci():
    import pandas as pd
    rows = [dict(sigma=2, contrast=.2, field_mm=1.0, true_density=d, est_density=e)
            for d, e in [(3, 3.0), (5, 5.0), (5, 5.1), (7, 7.0), (7, 7.2), (12, 1.0), (12, 0.0)]]
    r = W.s2_roc_reduced_ci(pd.DataFrame(rows), 2, .2, n_boot=50)
    assert r["pair"] == [5.0, 7.0] and r["auc"] == 1.0


def test_s7_reports_oracle_and_blind_icc():
    r = W.s7_repeatability(ScopeParams(), quick=True)
    assert "icc_oracle_position" in r and r["icc"] <= r["icc_oracle_position"] + 0.05


def test_pixel_area_scales_photons():
    sc = ScopeParams(um_per_px=1.0, sigma_um=0.5, illum_strength=0.0, photons=500, read_noise=0.0)
    sd = []
    for p in (1.0, 4.0):
        img, *_ = W.single_loop_image(sc, dict(limb_um=8, apical_um=8, total_um=30, limb_len=50.0), np.random.default_rng(1), pixel_um=p)
        sd.append(img[:6, :40].std())
    assert sd[1] < 0.6 * sd[0]                                 # 16x photons -> about 4x less relative noise


def test_load_gray_dtype_scaling_and_green(tmp_path):
    from skimage import io
    a = np.zeros((8, 8, 3), np.uint16); a[..., 1] = 257 * 100; a[..., 0] = 65535
    io.imsave(tmp_path / "a.tiff", a, check_contrast=False)
    assert intake.load_gray(tmp_path / "a.tiff").max() == pytest.approx(100.0)
    b = np.full((8, 8), 200, np.uint16)
    io.imsave(tmp_path / "b.tiff", b, check_contrast=False)
    assert intake.load_gray(tmp_path / "b.tiff").max() == pytest.approx(200 * 255 / 65535)
    assert intake.load_gray(tmp_path / "b.tiff", bits=12).max() == pytest.approx(200 * 255 / 4095)


def test_tilted_ruler_warns():
    rng = np.random.default_rng(0)
    img = np.ones((200, 300)); img[:, 20::20] = 0.3
    from scipy import ndimage as ndi
    img = ndi.rotate(img, 12, reshape=False, mode="nearest")
    with pytest.warns(RuntimeWarning):
        C.tick_profile(img)


def test_total_width_not_stretched_by_lighting_at_low_contrast():
    """Regression (2026-10-21 session settings): contrast 0.095 with a 0.156 lighting gradient made the old
    global-threshold total width read 99 um for a 40 um loop."""
    sc = ScopeParams(um_per_px=1.6, sigma_um=3.0, contrast=0.095, illum_strength=0.156, photons=50000, read_noise=0.0026)
    rng = np.random.default_rng(0)
    for tot in (30, 40, 45):
        img, px, cx, roi, _ = W.single_loop_image(sc, dict(limb_um=12, apical_um=12, total_um=tot, limb_len=100.0), rng)
        assert M.total_width_fwhm(img, px, roi) == pytest.approx(tot, rel=0.05)


def test_shared_sigma_fit_recovers_blur_and_lumen():
    from capsim import realdata as RD
    x = np.arange(21, dtype=float)
    rng = np.random.default_rng(0)
    profs, cents = [], []
    for w in (5.0, 6.0, 7.0, 8.0):
        p = 1.0 - 0.1 * RD._band(x, 10.0, w, 1.9) + rng.normal(0, 0.001, x.size)
        profs.append(p); cents.append(10.0)
    r = RD.fit_limbs_shared_sigma(profs, cents)
    assert r["sigma_px"] == pytest.approx(1.9, rel=0.1)
    assert np.allclose(r["widths_px"], [5, 6, 7, 8], atol=0.6)
    assert r["contrast"] == pytest.approx(0.1, rel=0.1)
