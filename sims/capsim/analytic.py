"""Closed-form reference for S1's pass test: half-depth width of a Gaussian-blurred bar."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf


def bar_profile(x, w, sigma):
    s = max(sigma, 1e-9) * np.sqrt(2)
    return 0.5 * (erf((x + w / 2) / s) - erf((x - w / 2) / s))


def blurred_bar_fwhm(w, sigma):
    """Width at half of the peak height of a bar of width w blurred by a Gaussian of sigma."""
    if sigma <= 0:
        return float(w)
    peak = bar_profile(0.0, w, sigma)
    f = lambda x: bar_profile(x, w, sigma) - 0.5 * peak
    return float(2 * brentq(f, 0.0, w / 2 + 12 * sigma))
