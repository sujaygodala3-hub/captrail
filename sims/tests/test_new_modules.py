import numpy as np
import pytest

from capsim import analytic, geometry as G, measure as M, optics as O, propagate as P, stats as S

RES = 0.5


def test_analytic_limits():
    assert analytic.blurred_bar_fwhm(12, 0) == 12
    assert analytic.blurred_bar_fwhm(12, 0.2) == pytest.approx(12, rel=0.01)
    assert analytic.blurred_bar_fwhm(8, 6) > 8


def test_make_loop_ground_truth():
    sp = G.make_loop(12, 45, 40)
    assert G.true_apical_width(sp) == pytest.approx(45, abs=0.5)
    assert G.true_total_width(sp) == pytest.approx(40)
    m = G.render_mask([G.LoopSpec(**{**sp.__dict__, "cx": 80, "y_base": 10})], (160, 220), RES)
    assert m[:, 160].sum() * RES == pytest.approx(45, abs=1.5)      # bulge wider than loop radius is rendered exactly


def test_loop_row_density():
    rng = np.random.default_rng(0)
    specs, W = G.loop_row(7, 1000 / 7, rng)
    assert len(specs) == 7 and W == pytest.approx(1000)


def test_apical_and_total_width_noise_free():
    sp = G.make_loop(12, 45, 40, limb_len=100.0)
    sp.cx, sp.y_base = 80, 10
    img = O.forward(G.render_mask([sp], (160, 220), RES), RES, sigma_um=1.0, pixel_um=1.0, contrast=0.35, bits=16)
    assert M.apical_width_fwhm(img, 1.0, 80) == pytest.approx(45, rel=0.05)
    assert M.total_width_fwhm(img, 1.0, (150, 190)) == pytest.approx(40, rel=0.05)
    assert np.nanmean(M.limb_widths_fixed(img, 1.0, (150, 190))) == pytest.approx(12, rel=0.25)


def test_fixed_threshold_is_biased_at_low_contrast_but_half_depth_is_not():
    sp = G.make_loop(12, 12, 40, limb_len=100.0)
    sp.cx, sp.y_base = 80, 10
    img = O.forward(G.render_mask([sp], (160, 220), RES), RES, sigma_um=1.0, pixel_um=1.0, contrast=0.08, bits=16)
    fx = np.nanmean(M.limb_widths_fixed(img, 1.0, (150, 190)))
    hd = np.nanmean(M.limb_widths_fwhm(img, 1.0, (150, 190)))
    assert abs(hd - 12) < abs(fx - 12) or np.isnan(fx)


def test_apex_count_clean_row():
    rng = np.random.default_rng(0)
    for n in (5, 7, 12):
        specs, W = G.loop_row(n, 1000 / n, rng, limb_gap=40, limb_len=100, width_art=12, width_ven=12)
        img = O.forward(G.render_mask(specs, (W, 220), RES), RES, sigma_um=2.0, pixel_um=1.0, contrast=0.35, bits=16)
        assert M.count_apices(img, 1.0) == n
        assert M.density_per_mm(img, 1.0) == pytest.approx(n, rel=0.01)


def test_quality_gate_catches_clipped_and_dark():
    ok = np.full((100, 100), 128.0) + np.random.default_rng(0).normal(0, 5, (100, 100))
    assert M.quality_gate(ok)["pass"]
    assert not M.quality_gate(np.full((100, 100), 255.0))["pass"]
    assert not M.quality_gate(np.full((100, 100), 10.0))["pass"]


def test_texture_and_shift_run():
    m = np.zeros((200, 300), bool)
    a = O.forward(m, RES, texture_amp=0.05, rng=np.random.default_rng(1), pixel_um=1.0, bits=16)
    assert a.std() > 0
    sp = G.make_loop(12, 12, 40, limb_len=100.0); sp.cx, sp.y_base = 80, 10
    mk = G.render_mask([sp], (160, 220), RES)
    b = O.forward(mk, RES, pixel_um=1.0, shift_um=(4.0, 0.0), bits=16)
    c = O.forward(mk, RES, pixel_um=1.0, bits=16)
    assert np.abs(b - c).max() > 0.05


def test_depth_model_monotone():
    c0, s0 = O.depth_model(0.5, 2.0, 0, 0.01, 0.02)
    c1, s1 = O.depth_model(0.5, 2.0, 100, 0.01, 0.02)
    assert c0 == 0.5 and s0 == 2.0 and c1 < c0 and s1 > s0


def test_shear_baseline_matches_plan_numbers():
    lo, hi = P.shear_range(12.0)
    assert lo == pytest.approx(0.24, abs=0.005) and hi == pytest.approx(2.4, abs=0.01)


def test_propagation_exponents():
    assert P.fractional_error_pct(12.0 * 1.10, 12.0, "resistance") == pytest.approx((1.1 ** -4 - 1) * 100)
    assert P.linear_amplification("flow") == -3 and P.linear_amplification("pressure") == 1
    mc = P.monte_carlo(12.0, [12.5, 13.0, 12.8, np.nan])
    assert mc["resistance"]["mean"] < mc["velocity"]["mean"] < 0 < mc["pressure"]["mean"]


def test_pries_secomb_is_not_silently_guessed():
    with pytest.raises(NotImplementedError):
        P.pries_secomb_viscosity(12.0, 0.45)


def test_bland_altman_and_repeatability():
    a = np.array([10, 12, 14, 16, 18.0])
    ba = S.bland_altman(a + 1.0, a)
    assert ba["bias"] == pytest.approx(1.0) and ba["sd"] == pytest.approx(0.0) and S.inside_limits(1.0, ba)
    r = S.repeatability([10, 10.5, 9.5, 10.2])
    assert r["rc"] == pytest.approx(1.96 * np.sqrt(2) * r["sd"])


def test_icc_known_cases():
    rng = np.random.default_rng(0)
    subj = rng.normal(15, 3, 40)
    good = np.c_[subj + rng.normal(0, 0.3, 40), subj + rng.normal(0, 0.3, 40)]
    bad = np.c_[subj + rng.normal(0, 6, 40), subj + rng.normal(0, 6, 40)]
    assert S.icc(good) > 0.95 > 0.5 > S.icc(bad)
    shifted = np.c_[subj, subj + 5.0]
    assert S.icc(shifted, "3,1") > 0.99 > S.icc(shifted, "2,1")      # absolute agreement punishes the offset
    lo, hi = S.icc_ci(good, n_boot=200)
    assert lo < S.icc(good) < hi + 1e-9


def test_roc_auc():
    assert S.roc_auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert S.roc_auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    lo, hi = S.roc_auc_ci(np.r_[np.random.default_rng(0).normal(0, 1, 50), np.random.default_rng(1).normal(1.5, 1, 50)], [0] * 50 + [1] * 50, n_boot=200)
    assert 0.6 < lo < hi <= 1.0
