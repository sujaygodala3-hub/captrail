"""Estimate the real microscope's PSF sigma from a calibration-slide line profile.

Bridge between simulation and hardware: image a tick mark on the 10 um slide,
take a 1D intensity profile across it, and fit  bg - A * (box(line_width) * Gauss(sigma)).
Feed the fitted sigma (and um/px from the scan page) into the sweeps.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


def _model(x_um, bg, amp, x0, line_um, sigma_um, pixel_um):
    # Gaussian-blurred box, integrated over each detector pixel (3-point sub-sampling).
    def blurred(x):
        s = max(sigma_um, 1e-3) * np.sqrt(2)
        a, b = x - x0 - line_um / 2, x - x0 + line_um / 2
        return 0.5 * (erf(-a / s) - erf(-b / s))
    off = np.array([-1, 0, 1]) * pixel_um / 3
    return bg - amp * np.mean([blurred(x_um + o) for o in off], axis=0)


def fit_sigma(profile, pixel_um, line_um):
    """Fit sigma (um) given a 1D profile across one dark line of known width `line_um`.

    Returns (sigma_um, fit_dict). Use the slide's real tick width; if unknown, treat the
    result as an upper bound on blur.
    """
    p = np.asarray(profile, float)
    x = (np.arange(len(p)) + 0.5) * pixel_um
    bg0, amp0, x00 = np.percentile(p, 90), np.percentile(p, 90) - p.min(), x[np.argmin(p)]

    def resid(q):
        return _model(x, q[0], q[1], q[2], line_um, q[3], pixel_um) - p

    r = least_squares(resid, [bg0, amp0, x00, 1.0 * pixel_um + 0.5],
                      bounds=([0, 0, x.min(), 0.05], [np.inf, np.inf, x.max(), 20 * pixel_um + 20]))
    return float(r.x[3]), dict(bg=r.x[0], amp=r.x[1], x0=r.x[2], rmse=float(np.sqrt(np.mean(r.fun ** 2))))
