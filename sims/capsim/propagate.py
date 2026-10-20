"""S6: push a width-measurement error through Poiseuille resistance and wall shear.

This is an AMPLIFICATION argument for an idealized straight rigid tube, not a prediction of
capillary blood flow. Real capillaries have red cells about as wide as the lumen, a
glycocalyx, and non-Newtonian behavior.

Held-fixed assumptions. Derivations: R = 8*mu*L/(pi*r^4); tau = 4*mu*Q/(pi*r^3);
with v = Q/(pi*r^2) that is tau = 4*mu*v/r = 8*mu*v/d; and tau = r*dP/(2L).

    resistance          R   ~ r^-4
    fixed flow rate Q   tau ~ r^-3
    fixed velocity v    tau ~ r^-1
    fixed pressure dP/L tau ~ r^+1
"""
from __future__ import annotations

import numpy as np

EXPONENT = {"resistance": -4.0, "flow": -3.0, "velocity": -1.0, "pressure": +1.0}


def baseline_shear_pa(mu_mpas, v_mm_s, d_um):
    """Wall shear tau = 8*mu*v/d for Poiseuille flow (own calculation).
    mu in mPa*s, v in mm/s, d in um -> Pa. Check: 12 um, 0.3 mm/s, 1.2 mPa*s gives 0.24 Pa."""
    return 8.0 * np.asarray(mu_mpas) * np.asarray(v_mm_s) / np.asarray(d_um)


def shear_range(d_um=12.0, v_mm_s=(0.3, 1.2), mu_mpas=(1.2, 3.0)):
    lo = baseline_shear_pa(mu_mpas[0], v_mm_s[0], d_um)
    hi = baseline_shear_pa(mu_mpas[1], v_mm_s[1], d_um)
    return float(lo), float(hi)


def fractional_error_pct(w_meas, w_true, assumption):
    """Percent error of the derived quantity when the measured width is w_meas, not w_true."""
    rel = np.asarray(w_meas, float) / np.asarray(w_true, float)
    return (rel ** EXPONENT[assumption] - 1.0) * 100.0


def linear_amplification(assumption):
    """Small-error amplification factor: d(Q)/Q = n * d(w)/w."""
    return EXPONENT[assumption]


def monte_carlo(w_true, w_meas_samples):
    """Summarize each assumption's error distribution from simulated width measurements.

    w_meas_samples: array of measured widths (um) for one true width (um), from S1.
    Returns {assumption: dict(mean, sd, p5, p95, median)} in percent.
    """
    w = np.asarray(w_meas_samples, float)
    w = w[~np.isnan(w)]
    if len(w) == 0:      # estimator failed on every sample (unresolved limbs): report NaN, not an error
        return {name: dict(mean=float('nan'), sd=float('nan'), p5=float('nan'), median=float('nan'), p95=float('nan')) for name in EXPONENT}
    out = {}
    for name in EXPONENT:
        e = fractional_error_pct(w, w_true, name)
        out[name] = dict(mean=float(e.mean()), sd=float(e.std(ddof=1)) if len(e) > 1 else float("nan"),
                         p5=float(np.percentile(e, 5)), median=float(np.median(e)), p95=float(np.percentile(e, 95)))
    return out


def pries_secomb_viscosity(*args, **kwargs):
    """Second-pass viscosity: Pries and Secomb 2005 in vivo law.

    NOT IMPLEMENTED ON PURPOSE. The plan says to implement it from the paper, not from memory,
    and the paper (Am J Physiol Heart Circ Physiol 289:H2657, doi 10.1152/ajpheart.00297.2005)
    was not readable when this was written. Read Eqs. for mu_0.45(D), C(D) and the relative
    apparent viscosity, then fill this in and add a test against a value printed in the paper.
    """
    raise NotImplementedError(pries_secomb_viscosity.__doc__)
