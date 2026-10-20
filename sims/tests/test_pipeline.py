import numpy as np
import pytest

from capsim import geometry as G, measure as M, optics as O, psf_fit as P

RES = 0.5


def one_loop(w, gap=60.0):
    sp = G.LoopSpec(cx=80, y_base=10, limb_gap=gap, limb_len=150, width_art=w, width_ven=w)
    return G.render_mask([sp], (160, 220), RES)


def roi(px=1.0):
    return int(60 / px), int(140 / px)  # rows in the straight limb section


def test_mask_area_matches_geometry():
    m = one_loop(12)
    straight_area = m[int(60 / RES):int(140 / RES)].sum() * RES ** 2
    assert straight_area == pytest.approx(2 * 12 * 80, rel=0.03)


def test_no_blur_recovers_width():
    img = O.forward(one_loop(12), RES, sigma_um=0.0, pixel_um=1.0)
    w = M.limb_widths_fwhm(img, 1.0, roi())
    assert np.nanmean(w) == pytest.approx(12, abs=1.0)


def test_heavy_blur_overestimates_thin_vessel():
    img = O.forward(one_loop(8), RES, sigma_um=6.0, pixel_um=1.0)
    assert np.nanmean(M.limb_widths_fwhm(img, 1.0, roi())) > 10


def test_merged_limbs_are_reported_not_hidden():
    img = O.forward(one_loop(12, gap=14), RES, sigma_um=8.0, pixel_um=1.0)
    assert np.isnan(M.limb_widths_fwhm(img, 1.0, roi())).any()


def test_count_clean_array():
    rng = np.random.default_rng(1)
    specs, W = G.loop_array(7, 120, rng, limb_gap=40, limb_len=100)
    img = O.forward(G.render_mask(specs, (W, 220), RES), RES, sigma_um=2.0)
    assert M.count_loops(img, 1.0) == 7


def test_psf_fit_recovers_sigma():
    x = (np.arange(60) + 0.5) * 0.8
    prof = P._model(x, 0.8, 0.3, 24.0, 5.0, 2.5, 0.8)
    prof = prof + np.random.default_rng(0).normal(0, 0.003, prof.shape)
    s, _ = P.fit_sigma(prof, 0.8, 5.0)
    assert s == pytest.approx(2.5, rel=0.1)
