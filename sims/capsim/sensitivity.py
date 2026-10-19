"""Which scope numbers matter, and how uncertain are the conclusions?

Two tools, both driven by ScopeParams:

* tornado():     vary each scope value by +-rel one at a time and rank how much a chosen
                 metric moves. Run it BEFORE S0 to learn what must be measured carefully.
* monte_carlo(): draw scope values from their S0 uncertainty (or a wide spread for values still
                 assumed), recompute the metric, and report a 95% interval. This turns
                 "sigma = 2.1" into "the 45 vs 55 um call error is 2% to 9%".

Metrics are small, fast versions of the S1 and S4 outputs so hundreds of draws stay cheap.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from . import measure as M
from .scope import FIELDS, ScopeParams
from .sweeps import GIANT_UM, _nanmean, single_loop_image

POSITIVE = {"um_per_px": 0.25, "sigma_um": 0.1, "photons": 50.0, "read_noise": 1e-4,
            "contrast": 0.02, "contrast_oil": 0.02}
MIN_ILLUM = 0.0


def metric_limb_bias(scope, w=12.0, reps=6, seed=0):
    """Percent bias of the half-depth limb width for a limb of width w um."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        img, px, cx, roi, _ = single_loop_image(scope, dict(limb_um=w, apical_um=w, total_um=w + 28, limb_len=100.0), rng)
        vals.append(_nanmean(M.limb_widths_fwhm(img, px, roi)))
    m = _nanmean(vals)
    return 100 * (m / w - 1) if not np.isnan(m) else float("nan")


def metric_giant_call_error(scope, reps=20, seed=0, lo=45.0, hi=55.0):
    """Combined error rate: P(45 um read as >= 50) and P(55 um read as < 50), averaged."""
    rng = np.random.default_rng(seed)
    def run(t):
        out = []
        for _ in range(reps):
            img, px, cx, _roi, _ = single_loop_image(scope, dict(limb_um=12, apical_um=t, total_um=40, limb_len=100.0), rng)
            out.append(M.apical_width_fwhm(img, px, cx))
        return np.array(out)
    a, b = run(lo), run(hi)
    return float(0.5 * ((a >= GIANT_UM).mean() + (b < GIANT_UM).mean()))


METRICS = {"limb12_bias_pct": metric_limb_bias, "giant_call_error": metric_giant_call_error}


def _set(scope: ScopeParams, name, value):
    s = copy.deepcopy(scope)
    setattr(s, name, value)
    return s


def tornado(scope, metric="limb12_bias_pct", rel=0.25, params=None):
    """One-at-a-time +-rel sweep. Returns a list of dicts sorted by |swing|, largest first."""
    fn = METRICS[metric] if isinstance(metric, str) else metric
    base = fn(scope)
    rows = []
    for name in params or ("um_per_px", "sigma_um", "photons", "read_noise", "illum_strength", "contrast"):
        v = getattr(scope, name)
        if not v:
            continue
        lo = fn(_set(scope, name, v * (1 - rel)))
        hi = fn(_set(scope, name, v * (1 + rel)))
        rows.append(dict(param=name, value=v, measured=bool(scope.measured.get(name)), low=lo, high=hi,
                         swing=abs(hi - lo), base=base))
    return sorted(rows, key=lambda r: -np.nan_to_num(r["swing"]))


def sample_scope(scope: ScopeParams, rng, rel_sd_assumed=0.25):
    """Draw one scope. Measured values use scope.uncertainty (absolute SD) if present, else are held;
    values still ASSUMED get a relative SD of `rel_sd_assumed`."""
    unc = getattr(scope, "uncertainty", {}) or {}
    s = copy.deepcopy(scope)
    for name in FIELDS:
        v = getattr(s, name)
        if v is None or name in ("illum_angle_deg",):
            continue
        sd = unc.get(name)
        if sd is None:
            sd = 0.0 if scope.measured.get(name) else abs(v) * rel_sd_assumed
        if sd > 0:
            v = float(rng.normal(v, sd))
            v = max(v, POSITIVE.get(name, MIN_ILLUM))
            setattr(s, name, v)
    return s


def monte_carlo(scope, metric="limb12_bias_pct", n=40, seed=0, rel_sd_assumed=0.25):
    """Distribution of a metric over scope uncertainty. Returns dict(values, mean, lo95, hi95, n_valid)."""
    fn = METRICS[metric] if isinstance(metric, str) else metric
    rng = np.random.default_rng(seed)
    vals = np.array([fn(sample_scope(scope, rng, rel_sd_assumed), seed=int(rng.integers(1 << 30))) for _ in range(n)], float)
    ok = vals[~np.isnan(vals)]
    return dict(values=vals, n_valid=int(len(ok)), mean=float(ok.mean()) if len(ok) else float("nan"),
                lo95=float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                hi95=float(np.percentile(ok, 97.5)) if len(ok) else float("nan"))
