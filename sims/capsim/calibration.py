"""S0: turn real calibration captures into scope parameters.

Four measurements, each from one kind of capture (see the imaging guide):

1. um per pixel   from the 10 um slide ticks             -> umpx_from_ticks
2. PSF sigma      from one tick or an edge profile        -> sigma_from_tick, sigma_from_edge
3. gain, read noise from blank-card frame pairs           -> photon_transfer (standard
                    photon-transfer method; the EMVA 1288 standard uses the same idea)
4. lighting gradient from a blank-card image              -> illumination_gradient

Images are 2D arrays in 8-bit digital numbers (0 to 255). If the camera applies gamma,
auto exposure or sharpening, photon transfer is not linear: the code reports r2 and a
`linear` flag. Lock exposure and white balance before capturing S0 frames.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import erf

from .psf_fit import fit_sigma


# --------------------------------------------------------------------------- um / px
def tick_profile(img, angles=np.linspace(-5, 5, 21)):
    """Average an image across tick direction. Tries small rotations and keeps the one
    with the sharpest ticks (largest profile variance). Ticks are assumed roughly vertical.
    Warns if the best angle is at the edge of the search (ticks tilted more than that: re-shoot or
    straighten the ruler, because um/px and sigma would be wrong)."""
    import warnings
    img = np.asarray(img, float)
    best, best_v, best_a = None, -1, 0.0
    angles = list(angles)
    for a in angles:
        r = ndi.rotate(img, a, reshape=False, order=3, mode="nearest") if a else img
        h, w = r.shape
        p = r[h // 4: 3 * h // 4].mean(0)
        v = p.var()
        if v > best_v:
            best, best_v, best_a = p, v, a
    if len(angles) > 2 and best_a in (min(angles), max(angles)):
        warnings.warn(f"ruler tilt at the edge of the search ({best_a:.1f} deg): calibration may be wrong", RuntimeWarning)
    return best


def detrend(p, deg=2, iters=4):
    """Divide out a slow lighting trend from a ruler profile using an asymmetric polynomial fit
    that follows the bright background (dark ticks are ignored). Returns profile / trend, mean 1."""
    p = np.asarray(p, float)
    x = np.linspace(-1, 1, len(p))
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        c = np.polyfit(x[keep], p[keep], deg)
        fit = np.polyval(c, x)
        resid = p - fit
        sd = resid[keep].std()
        keep = resid > -0.5 * sd if sd > 0 else keep
        if keep.sum() < deg + 3:
            break
    return p / np.maximum(np.polyval(np.polyfit(x[keep], p[keep], deg), x), 1e-9)


def _tick_peaks(p, pixel_scale_smooth=1.0):
    """Indices of dark ticks in a (detrended) profile, with a noise-aware prominence floor."""
    d = detrend(p)
    y = ndi.gaussian_filter1d(d, pixel_scale_smooth)
    inv = 1.0 - y
    dip = np.percentile(inv, 99)
    bg = inv < 0.25 * dip                      # estimate noise on flat background only: ticks inflate diffs
    use = bg[1:] & bg[:-1]
    dd = np.diff(d)[use] if use.sum() > 10 else np.diff(d)
    noise = np.median(np.abs(dd)) / 0.6745 / np.sqrt(2)
    pk, _ = find_peaks(inv, prominence=max(0.3 * dip, 4 * noise))
    return pk, inv


def _subpixel_peak(y, i):
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        d = a - 2 * b + c
        if d != 0:
            return i + 0.5 * (a - c) / d
    return float(i)


def umpx_from_ticks(profile, pitch_um=10.0, dark_ticks=True, min_ticks=4):
    """um per pixel from a 1D profile across a ruler with known tick pitch.

    Fits tick position = a + b * index. Returns (um_per_px, sd_of_slope_um_per_px, n_ticks).
    """
    p = np.asarray(profile, float)
    if not dark_ticks:
        p = p.max() + p.min() - p            # bright ticks: flip so ticks are dips
    pk, y = _tick_peaks(p)
    if len(pk) < min_ticks:
        raise ValueError(f"found {len(pk)} ticks, need at least {min_ticks}: check focus, crop and orientation")
    # reject a tick pitch outlier (a missed tick would shift indices)
    gaps = np.diff(pk)
    med = np.median(gaps)
    pos = np.array([_subpixel_peak(y, i) for i in pk])
    idx = np.rint((pos - pos[0]) / med)
    A = np.c_[np.ones_like(idx), idx]
    coef, res, *_ = np.linalg.lstsq(A, pos, rcond=None)
    b = coef[1]
    resid = pos - A @ coef
    dof = max(1, len(pos) - 2)
    se = np.sqrt(resid @ resid / dof / ((idx - idx.mean()) ** 2).sum())
    return float(pitch_um / b), float(pitch_um * se / b ** 2), len(pos)


def umpx_summary(values):
    """Mean and SD over repeated slide captures (the plan asks for 10 across two sessions)."""
    v = np.asarray(values, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                cv_pct=float(100 * v.std(ddof=1) / v.mean()) if len(v) > 1 else float("nan"), n=len(v))


# --------------------------------------------------------------------------- PSF sigma
def sigma_from_tick(profile, pixel_um, line_um, window_um=40.0):
    """Fit sigma on the strongest tick of a profile (ticks must be isolated; for a 10 um ruler use
    sigma_from_ticks). `line_um` must be the slide's real tick width."""
    p = np.asarray(profile, float)
    c = int(np.argmin(ndi.gaussian_filter1d(p, 1.0)))
    h = max(6, int(window_um / 2 / pixel_um))
    seg = p[max(0, c - h): c + h + 1]
    return fit_sigma(seg, pixel_um, line_um)


def _ticks_model(x, bg, amp, x0, sigma, pitch, line_um, n, pixel_um):
    s = max(sigma, 1e-3) * np.sqrt(2)
    off = np.array([-1, 0, 1]) * pixel_um / 3
    out = 0.0
    for i in range(n):
        c = x0 + i * pitch
        out = out + np.mean([0.5 * (erf((x + o - c + line_um / 2) / s) - erf((x + o - c - line_um / 2) / s)) for o in off], axis=0)
    return bg - amp * out


def sigma_from_ticks(profile, pixel_um, pitch_um, line_um):
    """PSF sigma from ALL ticks of a ruler profile at once (shared sigma, known pitch and tick width).

    Use this rather than sigma_from_tick whenever neighbouring ticks overlap in the blur
    (pitch under about 8 sigma, true for a 10 um slide). `line_um` is the real tick width.
    Returns (sigma_um, dict). Pitch is fitted within +-5% to absorb um/px error.
    """
    p = np.asarray(profile, float)
    x = (np.arange(len(p)) + 0.5) * pixel_um
    pk, inv = _tick_peaks(p)
    if len(pk) < 3:
        raise ValueError("need at least 3 ticks")
    n = int(round((pk[-1] - pk[0]) * pixel_um / pitch_um)) + 1
    x0 = x[pk[0]]
    bg0, amp0 = np.percentile(p, 90), np.percentile(p, 90) - p.min()
    r = least_squares(lambda q: _ticks_model(x, q[0], q[1], q[2], q[3], q[4], line_um, n, pixel_um) - p,
                      [bg0, amp0, x0, 1.5 * pixel_um + 0.5, pitch_um],
                      bounds=([0, 0, x0 - pitch_um / 2, 0.05, 0.95 * pitch_um],
                              [np.inf, np.inf, x0 + pitch_um / 2, 20 * pixel_um + 20, 1.05 * pitch_um]))
    return float(r.x[3]), dict(bg=r.x[0], amp=r.x[1], pitch=r.x[4], n_ticks=n, rmse=float(np.sqrt(np.mean(r.fun ** 2))))


def _edge_model(x, lo, hi, x0, sigma, pixel_um):
    off = np.array([-1, 0, 1]) * pixel_um / 3
    return np.mean([lo + (hi - lo) * 0.5 * (1 + erf((x + o - x0) / (max(sigma, 1e-3) * np.sqrt(2)))) for o in off], axis=0)


def sigma_from_edge(profile, pixel_um):
    """Fit a Gaussian-blurred step (edge spread function) to a 1D profile across a sharp edge.
    The fitted sigma is the PSF sigma if the edge is truly sharp (razor edge, black on white card)."""
    p = np.asarray(profile, float)
    x = (np.arange(len(p)) + 0.5) * pixel_um
    lo, hi = np.percentile(p, 5), np.percentile(p, 95)
    x0 = x[np.argmin(np.abs(p - (lo + hi) / 2))]
    r = least_squares(lambda q: _edge_model(x, q[0], q[1], q[2], q[3], pixel_um) - p,
                      [lo, hi, x0, 1.5 * pixel_um],
                      bounds=([-np.inf, -np.inf, x.min(), 0.05], [np.inf, np.inf, x.max(), 20 * pixel_um + 20]))
    return float(r.x[3]), dict(lo=r.x[0], hi=r.x[1], x0=r.x[2], rmse=float(np.sqrt(np.mean(r.fun ** 2))))


# --------------------------------------------------------------------------- photon transfer
def photon_transfer(levels, bits=8, sat_frac=0.92, min_levels=3):
    """Camera conversion gain and read noise from blank-card frames.

    levels: dict {name: [frame, frame, ...]} with >= 2 frames per exposure level, all taken
    with exposure locked. Temporal variance comes from the difference of frame pairs, so
    fixed-pattern lighting cancels. Model: var_DN = g * mean_DN + read_var_DN.

    Returns dict(g, read_noise_dn, r2, linear, means, vars). `g` is DN per electron-equivalent.
    """
    q = 2 ** bits - 1
    means, vars_ = [], []
    for name, frames in levels.items():
        f = [np.asarray(x, float) for x in frames]
        if len(f) < 2:
            continue
        m = np.mean(f, axis=0)
        keep = (m > 0.01 * q) & (m < sat_frac * q)
        if keep.sum() < 100:
            continue
        v = [0.5 * np.var((f[i] - f[i + 1])[keep]) for i in range(0, len(f) - 1)]
        means.append(float(m[keep].mean()))
        vars_.append(float(np.mean(v)))
    if len(means) < min_levels:
        raise ValueError(f"need >= {min_levels} usable exposure levels, got {len(means)}")
    means, vars_ = np.array(means), np.array(vars_)
    A = np.c_[means, np.ones_like(means)]
    (g, c), *_ = np.linalg.lstsq(A, vars_, rcond=None)
    pred = A @ np.array([g, c])
    ss_res, ss_tot = ((vars_ - pred) ** 2).sum(), ((vars_ - vars_.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    q_noise = 1.0 / 12.0 if bits == 8 else (2 ** (8 - bits)) ** 2 / 12.0   # quantization noise in DN^2
    read_var = max(c - q_noise, 0.0)
    n = len(means)
    if n > 2:                                       # OLS standard errors of slope and intercept
        s2 = ss_res / (n - 2)
        cov = s2 * np.linalg.inv(A.T @ A)
        se_g, se_c = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    else:
        se_g = se_c = float("nan")
    se_rn = se_c / (2 * np.sqrt(max(read_var, 1e-9))) if np.isfinite(se_c) else float("nan")
    return dict(g=float(g), read_noise_dn=float(np.sqrt(read_var)), r2=float(r2), se_g=se_g, se_read_noise_dn=float(se_rn),
                linear=bool(r2 > 0.98 and g > 0), means=means.tolist(), vars=vars_.tolist())


def to_forward(g, read_noise_dn, bits=8):
    """Map photon-transfer results onto optics.forward(photons=, read_noise=)."""
    q = 2 ** bits - 1
    return dict(photons=float(q / g), read_noise=float(read_noise_dn / q))


def forward_sd(g, se_g, read_noise_dn, se_rn_dn, bits=8):
    """1-SD of the forward-model photons and read_noise from the photon-transfer standard errors."""
    q = 2 ** bits - 1
    return dict(photons=float(q * se_g / g ** 2) if np.isfinite(se_g) else None,
                read_noise=float(se_rn_dn / q) if np.isfinite(se_rn_dn) else None)


# --------------------------------------------------------------------------- lighting
def illumination_gradient(flat, smooth_um=None, pixel_um=1.0):
    """Planar lighting gradient of a blank-card image, in the same terms as optics.forward.

    Returns dict(strength = peak-to-peak plane variation / mean, angle_deg, residual_pct)
    where residual_pct is the remaining non-planar (vignetting) peak-to-peak variation.
    """
    f = np.asarray(flat, float)
    s = (smooth_um or 0.05 * max(f.shape) * pixel_um) / pixel_um
    f = ndi.gaussian_filter(f, s, mode="nearest")
    mean = f.mean()
    ny, nx = f.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.c_[xx.ravel(), yy.ravel(), np.ones(f.size)]
    (gx, gy, c), *_ = np.linalg.lstsq(A, f.ravel() / mean, rcond=None)
    strength = abs(gx) * (nx - 1) + abs(gy) * (ny - 1)
    plane = (A @ np.array([gx, gy, c])).reshape(f.shape)
    resid = (f / mean - plane)
    return dict(strength=float(strength), angle_deg=float(np.degrees(np.arctan2(gy, gx))),
                residual_pct=float(100 * (resid.max() - resid.min())))
