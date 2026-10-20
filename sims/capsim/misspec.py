"""S8: how wrong are the S1 to S4 predictions if the real blur is not Gaussian?

The forward model assumes a Gaussian PSF. Real scopes are closer to a defocus disc or have
long tails. This module builds "truth" images with a disc or exponential PSF, does what the real
workflow does (fit a Gaussian sigma to the slide ticks), then compares the width bias and giant-call
error in the truth images with what the Gaussian model predicts at the fitted sigma. The gap is the
model-misspecification error you should carry as extra uncertainty.

Kernels are scaled so every PSF has the same per-axis standard deviation sigma.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from . import calibration as C, geometry as G, measure as M, optics as O
from .scope import ScopeParams
from .sweeps import GIANT_UM, _fov, _nanmean, _roi

KINDS = ("gaussian", "disc", "laplace")


def psf_kernel(kind, sigma_um, res_um):
    """2D PSF with per-axis SD sigma_um on a res_um grid, sum 1."""
    ext = 12 * sigma_um if kind == "laplace" else 5 * sigma_um
    n = int(math.ceil(ext / res_um))
    ax = np.arange(-n, n + 1) * res_um
    X, Y = np.meshgrid(ax, ax)
    r = np.hypot(X, Y)
    if kind == "gaussian":
        k = np.exp(-0.5 * (r / sigma_um) ** 2)
    elif kind == "disc":                       # uniform disc of radius R
        R = 2.0 * sigma_um                     # its per-axis variance is R^2/4, so R = 2 sigma
        k = (r <= R).astype(float)
    elif kind == "laplace":                    # radial exponential exp(-r/b): per-axis variance 3 b^2
        k = np.exp(-r / (sigma_um / math.sqrt(3)))
    else:
        raise ValueError(kind)
    return k / k.sum()


def blur_mask(mask, res_um, kind, sigma_um):
    if sigma_um <= 0:
        return mask.astype(np.float32)
    return fftconvolve(mask.astype(np.float32), psf_kernel(kind, sigma_um, res_um), mode="same").astype(np.float32)


def truth_image(scope: ScopeParams, spec_kw, rng, kind, sigma_um, **over):
    """One loop imaged with a non-Gaussian PSF (forward model's own blur switched off)."""
    px = scope.um_per_px
    res = px / math.ceil(px / 0.5 - 1e-9)
    kw = scope.forward_kwargs(sigma_um=0.0, **over)
    W, H = _fov(px)
    spec = G.make_loop(**spec_kw)
    spec.cx, spec.y_base = W / 2, 10.0
    mask = G.render_mask([spec], (W, H), res)
    img = O.forward(blur_mask(mask, res, kind, sigma_um), res, rng=rng, **kw)
    return img, px, spec.cx, _roi(px, H)


def truth_slide_profile(scope, kind, sigma_um, pitch_um=10.0, tick_um=2.0, n_ticks=9, seed=0):
    px = scope.um_per_px
    res = px / math.ceil(px / 0.5 - 1e-9)
    W = math.ceil((n_ticks + 2) * pitch_um / px) * px
    H = math.ceil(60 / px) * px
    mask = np.zeros((int(round(H / res)), int(round(W / res))), bool)
    xs = (np.arange(mask.shape[1]) + 0.5) * res
    for i in range(n_ticks):
        mask[:, np.abs(xs - (i + 1.5) * pitch_um) <= tick_um / 2] = True
    img = O.forward(blur_mask(mask, res, kind, sigma_um), res, rng=np.random.default_rng(seed),
                    **scope.forward_kwargs(sigma_um=0.0, contrast=0.6))
    return img[img.shape[0] // 4: 3 * img.shape[0] // 4].mean(0)


def _giant_error(imgs_lo, imgs_hi):
    return 0.5 * (np.mean(np.array(imgs_lo) >= GIANT_UM) + np.mean(np.array(imgs_hi) < GIANT_UM))


def misspecification(scope: ScopeParams, kinds=("disc", "laplace"), sigmas=(2.0, 3.0, 4.0), reps=10, seed=8,
                     tick_um=2.0, pitch_um=10.0):
    """Rows: kind, sigma_true, sigma_fit (Gaussian fit to the slide), then limb and apical bias and giant-call
    error in the truth images vs the Gaussian model at sigma_fit, and their differences."""
    from .sweeps import single_loop_image
    rng = np.random.default_rng(seed)
    rows = []
    for kind in kinds:
        for st in sigmas:
            prof = truth_slide_profile(scope, kind, st, pitch_um, tick_um, seed=int(rng.integers(1 << 30)))
            try:
                sfit, _ = C.sigma_from_ticks(prof, scope.um_per_px, pitch_um, tick_um)
            except Exception:
                sfit = float("nan")
            res = {}
            for tag in ("truth", "model"):
                if tag == "model" and not np.isfinite(sfit):      # ruler unresolvable: no Gaussian sigma to predict from
                    res[tag] = dict(limb_bias=np.nan, apical45_bias=np.nan, giant_err=np.nan)
                    continue
                limb, ap_lo, ap_hi = [], [], []
                for _ in range(reps):
                    def one(spec_kw):
                        if tag == "truth":
                            img, px, cx, roi = truth_image(scope, spec_kw, rng, kind, st)
                        else:
                            img, px, cx, roi, _ = single_loop_image(scope, spec_kw, rng, sigma=sfit)
                        return img, px, cx, roi
                    img, px, cx, roi = one(dict(limb_um=12, apical_um=12, total_um=40, limb_len=100.0))
                    limb.append(_nanmean(M.limb_widths_fwhm(img, px, roi)))
                    img, px, cx, roi = one(dict(limb_um=12, apical_um=45, total_um=40, limb_len=100.0))
                    ap_lo.append(M.apical_width_fwhm(img, px, cx))
                    img, px, cx, roi = one(dict(limb_um=12, apical_um=55, total_um=40, limb_len=100.0))
                    ap_hi.append(M.apical_width_fwhm(img, px, cx))
                res[tag] = dict(limb_bias=100 * (_nanmean(limb) / 12 - 1), apical45_bias=100 * (np.nanmean(ap_lo) / 45 - 1),
                                giant_err=float(_giant_error(ap_lo, ap_hi)))
            rows.append(dict(psf=kind, sigma_true=st, sigma_gauss_fit=sfit,
                             limb_bias_truth=res["truth"]["limb_bias"], limb_bias_model=res["model"]["limb_bias"],
                             limb_bias_gap=res["model"]["limb_bias"] - res["truth"]["limb_bias"],
                             apical45_bias_truth=res["truth"]["apical45_bias"], apical45_bias_model=res["model"]["apical45_bias"],
                             giant_err_truth=res["truth"]["giant_err"], giant_err_model=res["model"]["giant_err"]))
    return pd.DataFrame(rows)
