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


def _tick_band_rows(img, pmin=8.0, pmax=40.0, frac=0.2):
    """Rows inside the fine-tick band: the rows whose profile is most periodic at any pitch in [pmin, pmax]."""
    g = np.asarray(img, float)
    x = np.arange(g.shape[1])
    rd = g - g.mean(1, keepdims=True)
    pw = np.zeros(g.shape[0])
    for pitch in np.arange(pmin, pmax, 0.1):
        pw = np.maximum(pw, np.abs(rd @ np.exp(-2j * np.pi * x / pitch)))
    return pw >= np.percentile(pw, 100 * (1 - frac))


def slide_pitch_px(img, angles=np.linspace(-3, 3, 13), pmin=8.0, pmax=40.0):
    """Fine-tick pitch of a stage micrometer image, in pixels, robust to the crosshair, long ticks, dust and
    split dips (which break peak-picking). Rows of the tick band only; the best small rotation; then the pitch
    from the AUTOCORRELATION of the band profile: lag of the k-th peak against k (k = 1..K), a straight line
    whose slope is the pitch. Returns dict(pitch_px, se_px, angle_deg, rows, n_periods, fft_pitch_px)."""
    from scipy.signal import find_peaks as _fp
    g0 = np.asarray(img, float)
    best = None
    for a in angles:
        g = ndi.rotate(g0, a, reshape=False, order=1, mode="nearest") if a else g0
        h, w = g.shape
        g = g[h // 10: h - h // 10, w // 10: w - w // 10]       # rotation corners
        keep = _tick_band_rows(g, pmin, pmax)
        band = g[keep].mean(0)
        d = band - ndi.uniform_filter1d(band, int(2 * pmax + 1))
        if best is None or d.var() > best[0]:
            best = (d.var(), a, d, int(keep.sum()))
    _, ang, d, nrows = best
    d = d[int(pmax):-int(pmax)]
    N = 1 << 16
    F = np.abs(np.fft.rfft(d * np.hanning(len(d)), N))
    fr = np.fft.rfftfreq(N)
    m = (fr > 1 / pmax) & (fr < 1 / pmin)
    p_fft = float(1 / fr[m][np.argmax(F[m])])
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    ac = ac / ac[0]
    K = max(2, int(0.5 * len(d) / p_fft))
    lags, ks = [], []
    for k in range(1, K + 1):
        lo, hi = int(round(k * p_fft - 0.4 * p_fft)), int(round(k * p_fft + 0.4 * p_fft))
        if hi + 1 >= len(ac):
            break
        i = lo + int(np.argmax(ac[lo:hi + 1]))
        lags.append(_subpixel_peak(ac, i))
        ks.append(k)
    ks, lags = np.array(ks, float), np.array(lags, float)
    A = np.c_[np.ones_like(ks), ks]
    coef, *_ = np.linalg.lstsq(A, lags, rcond=None)
    r = lags - A @ coef
    se = float(np.sqrt(r @ r / max(1, len(ks) - 2) / ((ks - ks.mean()) ** 2).sum())) if len(ks) > 2 else float("nan")
    ac_mean = float(np.mean([ac[int(round(l))] for l in lags[:5]])) if len(lags) else 0.0
    return dict(pitch_px=float(coef[1]), se_px=se, angle_deg=float(ang), rows=nrows, n_periods=int(len(ks)), fft_pitch_px=p_fft,
                autocorr=ac_mean)


def umpx_from_slide(img, pitch_um=10.0, **kw):
    """um per pixel from a stage-micrometer IMAGE (preferred over umpx_from_ticks on real photos).
    Returns (um_per_px, se_um_per_px, info)."""
    r = slide_pitch_px(img, **kw)
    return float(pitch_um / r["pitch_px"]), float(pitch_um * r["se_px"] / r["pitch_px"] ** 2), r


def long_tick_spacing_px(img, rows_keep=None, min_sep=60):
    """Spacing of the LONG ticks (rows just outside the fine-tick band, where only long ticks reach).
    On a standard micrometer it is 10 fine pitches: an independent check of slide_pitch_px."""
    g = np.asarray(img, float)
    keep = _tick_band_rows(g) if rows_keep is None else rows_keep
    rows = np.where(keep)[0]
    out = np.r_[np.arange(max(0, rows.min() - 60), max(0, rows.min() - 25)), np.arange(rows.max() + 25, min(g.shape[0], rows.max() + 60))]
    if len(out) < 5:
        return float("nan")
    lp = g[out].mean(0)
    ld = lp - ndi.uniform_filter1d(lp, 81)
    pk, _ = find_peaks(-ld, prominence=0.4 * np.percentile(-ld, 99), distance=min_sep)
    if len(pk) < 3:
        return float("nan")
    dd = np.diff(pk)
    dd = dd[np.abs(dd - np.median(dd)) < 0.15 * np.median(dd)]
    return float(np.mean(dd)) if len(dd) else float("nan")


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


# ---- blank-card noise spectrum (single exposure level) -------------------------------------------
NOISE_BANDS = ((0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 40.0))   # difference-of-Gaussian bands, px


def _mad(x):
    x = np.asarray(x, float).ravel()
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _dog(x, a, b):
    from scipy import ndimage as ndi
    return ndi.gaussian_filter(x, a, mode="reflect") - ndi.gaussian_filter(x, b, mode="reflect")


def blank_card_noise(frames, bands=NOISE_BANDS):
    """Temporal noise of repeated blank-card frames (2D green-channel arrays, same exposure, card not moved).

    Uses consecutive-frame differences, so fixed texture and vignetting cancel; robust (MAD) per band,
    divided by sqrt(2) to give the noise of ONE frame. Returns dict with per-band DN, the single-pixel
    high-pass noise (same estimator as realdata.quality), mean level, frame-to-frame brightness drift and
    the flat-field shape (centre/edge ratio, p2-p98 spread of the smoothed card)."""
    from scipy import ndimage as ndi
    G = [np.asarray(f, float) for f in frames]
    if len(G) < 2:
        raise ValueError("need >= 2 blank-card frames")
    diffs = [G[i] - G[i + 1] for i in range(len(G) - 1)]
    band_dn = {f"{a}-{b}": float(np.mean([_mad(_dog(d, a, b)) for d in diffs]) / np.sqrt(2)) for a, b in bands}
    hp = [_mad(g - ndi.gaussian_filter(g, 1.5)) for g in G]
    means = [float(g.mean()) for g in G]
    sm = ndi.gaussian_filter(np.mean(G, 0), 25)
    H, W = sm.shape
    c = sm[H // 2 - 40:H // 2 + 40, W // 2 - 40:W // 2 + 40].mean()
    e = np.mean([sm[:60, :60].mean(), sm[:60, -60:].mean(), sm[-60:, :60].mean(), sm[-60:, -60:].mean()])
    p2, p98 = np.percentile(sm, [2, 98])
    return dict(n_frames=len(G), band_dn=band_dn, pixel_hp_noise_dn=float(np.median(hp)), mean_dn=float(np.mean(means)),
                drift_dn=float(max(means) - min(means)), flat_centre_over_edge=float(c / e),
                flat_spread_frac=float((p98 - p2) / sm.mean()))


def fit_corr_noise(band_dn, white_dn, scales=(1.0, 2.0, 4.0, 8.0), shape=(480, 640), seed=0, bands=NOISE_BANDS):
    """Correlated-noise components that, ADDED to white noise of `white_dn` per pixel, reproduce the measured
    per-band noise. Variances add across independent components, so this is NNLS on band variances.
    Returns dict(components=[[scale_px, std_DN], ...], fitted_band_dn, measured_band_dn, white_band_dn)."""
    from scipy.optimize import nnls
    from capsim.optics import corr_field
    rng = np.random.default_rng(seed)
    keys = [f"{a}-{b}" for a, b in bands]
    obs = np.array([band_dn[k] for k in keys]) ** 2
    w = rng.normal(size=shape)
    white = np.array([_mad(_dog(w, a, b)) for a, b in bands]) * white_dn
    R = np.array([[_mad(_dog(corr_field(shape, s, rng), a, b)) for a, b in bands] for s in scales]) ** 2
    amp2, _ = nnls(R.T, np.clip(obs - white ** 2, 0, None))
    fit = np.sqrt(white ** 2 + R.T @ amp2)
    return dict(components=[[float(s), float(np.sqrt(a))] for s, a in zip(scales, amp2) if a > 0],
                fitted_band_dn=dict(zip(keys, map(float, fit))), measured_band_dn=dict(zip(keys, map(float, np.sqrt(obs)))),
                white_band_dn=dict(zip(keys, map(float, white))))


def white_from_pixel_noise(pixel_hp_dn, mean_dn, bits=8, shape=(480, 640), seed=0):
    """Photons (forward-model units) whose shot noise gives `pixel_hp_dn` after the realdata.quality high-pass,
    at a mean level of `mean_dn`. Read noise is folded in (single exposure level: they cannot be separated)."""
    from scipy import ndimage as ndi
    q = 2 ** bits - 1
    w = np.random.default_rng(seed).normal(size=shape)
    atten = _mad(w - ndi.gaussian_filter(w, 1.5))          # high-pass response of unit white noise
    sd_dn = pixel_hp_dn / atten
    level = mean_dn / q
    return dict(photons=float(level / (sd_dn / q) ** 2), white_dn=float(sd_dn), hp_attenuation=float(atten))


def tick_blur_sigma(img, pitch_px=None, bins_per_px=4):
    """Blur at the RULER's zoom. The band profile is folded on the tick pitch (all ticks averaged onto one
    period, sub-pixel), then fit as one dark box of unknown width blurred by a Gaussian (the same model as the
    capillary limbs). Folding uses every tick, so dust and split dips average out.
    Returns dict(sigma_px, tick_width_px, pitch_px, rows, n_ticks)."""
    from scipy.optimize import least_squares
    g = np.asarray(img, float)
    r0 = slide_pitch_px(g)
    if pitch_px is None:
        pitch_px = r0["pitch_px"]
    if r0["angle_deg"]:
        g = ndi.rotate(g, r0["angle_deg"], reshape=False, order=1, mode="nearest")
    h, w = g.shape
    g = g[h // 10: h - h // 10, w // 10: w - w // 10]
    keep = _tick_band_rows(g)
    band = g[keep].mean(0)
    band = band / ndi.uniform_filter1d(band, int(4 * pitch_px) | 1)
    x = np.arange(len(band), dtype=float)
    ph = (x % pitch_px)
    nb = int(round(pitch_px * bins_per_px))
    idx = np.minimum((ph / pitch_px * nb).astype(int), nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    prof = np.bincount(idx, weights=band, minlength=nb) / np.maximum(cnt, 1)
    ok = cnt > 0
    prof[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], prof[ok], period=nb)
    shift = nb // 2 - int(np.argmin(ndi.uniform_filter1d(prof, bins_per_px, mode="wrap")))
    prof, ok = np.roll(prof, shift), np.roll(ok, shift)
    xb = (np.arange(nb) + 0.5) / bins_per_px
    c0 = xb[nb // 2]

    def model(q, xx):
        bg, A, x0, w, s = q
        s2 = max(s, 1e-3) * np.sqrt(2)
        y = np.zeros_like(xx)
        for m in (-1, 0, 1):      # neighbouring ticks wrap into the period
            y += 0.5 * (erf((xx - x0 - m * pitch_px + w / 2) / s2) - erf((xx - x0 - m * pitch_px - w / 2) / s2))
        return bg - A * y

    q0 = [np.percentile(prof, 95), np.ptp(prof), c0, pitch_px / 2, 1.0]
    r = least_squares(lambda q: (model(q, xb) - prof)[ok], q0,
                      bounds=([0, 0, c0 - pitch_px / 4, 0.5, 0.05], [np.inf, np.inf, c0 + pitch_px / 4, pitch_px - 0.5, pitch_px / 2]))
    return dict(sigma_px=float(r.x[4]), tick_width_px=float(r.x[3]), pitch_px=float(pitch_px), rows=int(keep.sum()),
                n_ticks=int(len(band) / pitch_px), rmse=float(np.sqrt(np.mean(r.fun ** 2))))
