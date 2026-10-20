import numpy as np
import pytest

from capsim import analytic, calibration as C, optics as O, sweeps as W
from capsim.scope import ScopeParams


def test_umpx_recovered_from_simulated_slide():
    sc = ScopeParams(um_per_px=1.0, sigma_um=1.5, photons=4000, read_noise=0.005)
    _, prof = W.slide_profile_sim(sc, pitch_um=10.0, tick_um=2.0, n_ticks=9)
    v, se, n = C.umpx_from_ticks(prof, 10.0)
    assert v == pytest.approx(1.0, rel=0.02) and n >= 8


def test_umpx_needs_enough_ticks():
    with pytest.raises(ValueError):
        C.umpx_from_ticks(np.ones(200), 10.0)


def test_sigma_from_tick_recovers_blur():
    sc = ScopeParams(um_per_px=0.5, sigma_um=2.5, photons=20000, read_noise=0.002)
    _, prof = W.slide_profile_sim(sc, pitch_um=10.0, tick_um=2.0, n_ticks=9)
    s, fit = C.sigma_from_ticks(prof, 0.5, 10.0, 2.0)
    assert s == pytest.approx(2.5, rel=0.1)


def test_isolated_tick_fit_still_works_when_ticks_are_far_apart():
    sc = ScopeParams(um_per_px=0.5, sigma_um=1.5, photons=20000, read_noise=0.002)
    _, prof = W.slide_profile_sim(sc, pitch_um=40.0, tick_um=2.0, n_ticks=5)
    s, _ = C.sigma_from_tick(prof, 0.5, 2.0, window_um=60)
    assert s == pytest.approx(1.5, rel=0.15)


def test_sigma_from_edge_recovers_blur():
    x = (np.arange(80) + 0.5) * 0.5
    p = C._edge_model(x, 30, 200, 20.0, 2.0, 0.5) + np.random.default_rng(0).normal(0, 0.5, x.shape)
    s, _ = C.sigma_from_edge(p, 0.5)
    assert s == pytest.approx(2.0, rel=0.1)


def test_photon_transfer_recovers_gain_and_read_noise():
    rng = np.random.default_rng(0)
    g_true, rn_true = 0.25, 1.5            # DN per electron, read noise in DN
    levels = {}
    for name, L in (("a", 150.0), ("b", 300.0), ("c", 450.0), ("d", 600.0)):
        # electrons L (mean) -> DN = g*e + read
        levels[name] = [np.clip(np.round(g_true * rng.poisson(L, (120, 120)) + rng.normal(0, rn_true, (120, 120))), 0, 255)
                        for _ in range(4)]
    pt = C.photon_transfer(levels)
    assert pt["g"] == pytest.approx(g_true, rel=0.1)
    assert pt["read_noise_dn"] == pytest.approx(rn_true, rel=0.3)
    assert pt["linear"]
    kw = C.to_forward(pt["g"], pt["read_noise_dn"])
    assert kw["photons"] == pytest.approx(255 / g_true, rel=0.1)


def test_photon_transfer_flags_nonlinear_camera():
    rng = np.random.default_rng(1)
    levels = {n: [np.clip(255 * (rng.poisson(L, (100, 100)) / 700.0) ** 0.45, 0, 255).round() for _ in range(3)]
              for n, L in (("a", 50), ("b", 150), ("c", 300), ("d", 500))}
    pt = C.photon_transfer(levels)
    assert not pt["linear"]


def test_photon_transfer_needs_levels():
    with pytest.raises(ValueError):
        C.photon_transfer({"a": [np.full((50, 50), 100.0)] * 2})


def test_illumination_gradient_recovers_strength_and_angle():
    flat = O.forward(np.zeros((400, 600), bool), 1.0, sigma_um=0, contrast=0.0, illum_strength=0.2, illum_angle_deg=20,
                     pixel_um=1.0, exposure=0.8, bits=16)
    g = C.illumination_gradient(flat)
    assert g["strength"] == pytest.approx(0.2, rel=0.2)
    assert g["angle_deg"] == pytest.approx(20, abs=5)


def test_scope_params_roundtrip_and_flags(tmp_path):
    sc = ScopeParams()
    assert "sigma_um" in sc.assumed()
    sc.mark(sigma_um=3.1, um_per_px=2.0)
    p = tmp_path / "s.json"
    sc.save(p)
    b = ScopeParams.load(p)
    assert b.sigma_um == 3.1 and b.measured["sigma_um"] and "sigma_um" not in b.assumed()
    assert b.tag().startswith("assumed")
    assert 2.0 / b.fine_res() == pytest.approx(round(2.0 / b.fine_res()))
    with pytest.raises(KeyError):
        sc.mark(nonsense=1)


def test_half_depth_matches_closed_form_bar():
    """S1 pass test: simulated half-depth width of a blurred bar is within 2% of the closed form."""
    rng = np.random.default_rng(0)
    for w, s in ((12, 3.0), (20, 4.0), (12, 4.0)):
        img, px, cx, roi, _ = W.single_loop_image(
            ScopeParams(um_per_px=0.5, sigma_um=s, photons=None, read_noise=0.0, illum_strength=0.0, contrast=0.35),
            dict(limb_um=w, apical_um=w, total_um=w + 60, limb_len=100.0), rng)
        from capsim import measure as M
        sim = np.nanmean(M.limb_widths_fwhm(img, px, roi, flatfield=False))
        assert sim == pytest.approx(analytic.blurred_bar_fwhm(w, s), rel=0.02)
