"""Helpers for real Celestron / scan-page frames: sort raw vs processed files, mask specular glare,
measure image quality, and estimate the blur (PSF sigma) directly from the capillary limbs.

Everything here works in PIXELS. Converting to micrometers needs um/px from the ruler (S0).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.special import erf

ENHANCED_MEAN = np.array([200.0, 120.0, 100.0])   # the scan page's "Enhance: Capillary" output is normalised to about this colour


def classify(rgb):
    """'grey' (single channel saved as RGB), 'enhanced' (scan page's processed view) or 'raw'."""
    a = np.asarray(rgb, float)
    if np.allclose(a[..., 0], a[..., 1]) and np.allclose(a[..., 1], a[..., 2]):
        return "grey"
    m = a.reshape(-1, 3).mean(0)
    return "enhanced" if np.all(np.abs(m - ENHANCED_MEAN) < 10) else "raw"


def glare_mask(rgb, thr=215.0, grow_px=6):
    """Specular reflections (LED glare on the oil film and bubbles): bright, near-white pixels, grown by grow_px."""
    a = np.asarray(rgb, float)
    m = (a.mean(2) > thr) | (a.min(2) > 0.80 * 255)
    m = ndi.binary_opening(m, iterations=1)
    return ndi.binary_dilation(m, iterations=grow_px) if grow_px else m


def clean_green(rgb, mask=None, radius=7):
    """Green channel (best loop contrast: hemoglobin absorbs green) with glare inpainted. Float 0..1."""
    import cv2
    a = np.asarray(rgb)
    g = a[..., 1].astype(np.uint8)
    if mask is None:
        mask = glare_mask(a)
    if mask.any():
        g = cv2.inpaint(g, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return g.astype(float) / 255.0


def quality(rgb, mask=None):
    """Per-frame quality numbers (computed on the non-glare area of the green channel)."""
    import cv2
    a = np.asarray(rgb, float)
    if mask is None:
        mask = glare_mask(rgb)
    ok = ~ndi.binary_dilation(mask, iterations=4)
    g = a[..., 1].astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    v = hp[ok]
    return dict(glare_frac=float(mask.mean()), clipped_frac=float((a >= 250).any(2).mean()),
                sharpness=float(lap[ok].var()) if ok.any() else float("nan"),
                noise_dn=float(1.4826 * np.median(np.abs(v - np.median(v)))) if ok.any() else float("nan"),
                mean_r=float(a[..., 0][ok].mean()), mean_g=float(a[..., 1][ok].mean()), mean_b=float(a[..., 2][ok].mean()))


def _band(x, x0, w, s):
    s = max(s, 1e-3) * np.sqrt(2)
    return 0.5 * (erf((x - x0 + w / 2) / s) - erf((x - x0 - w / 2) / s))


def fit_limbs_shared_sigma(profiles, centres, max_w=30.0):
    """Fit dark bands (box of width w blurred by a Gaussian of sigma) to several limb profiles at once,
    with ONE sigma shared by all of them (same optics, same frame) and its own width, depth and position
    per limb. Sharing sigma is what separates blur from true width (Gaussian blur of a narrow box looks
    like a wider box otherwise). Profiles are 1D arrays in pixels, centres the rough dip positions.

    Returns dict(sigma_px, widths_px, rmse, n). Widths here are LUMEN widths after removing blur."""
    n = len(profiles)
    if n < 2:
        return dict(sigma_px=float("nan"), widths_px=[], rmse=float("nan"), n=n)
    xs = [np.arange(len(p), dtype=float) for p in profiles]
    p0 = [2.0]
    lo, hi = [0.3], [15.0]
    for p, c in zip(profiles, centres):
        bg = np.percentile(p, 90)
        p0 += [bg, max(bg - p.min(), 1e-3), float(c), 6.0]
        lo += [0, 0, c - 6, 0.5]
        hi += [np.inf, np.inf, c + 6, max_w]

    def resid(q):
        s = q[0]
        out = []
        for i, (x, p) in enumerate(zip(xs, profiles)):
            bg, A, x0, w = q[1 + 4 * i: 5 + 4 * i]
            out.append(bg - A * _band(x, x0, w, s) - p)
        return np.concatenate(out)

    r = least_squares(resid, p0, bounds=(lo, hi))
    widths = [float(r.x[4 + 4 * i]) for i in range(n)]
    amps = [float(r.x[2 + 4 * i] / max(r.x[1 + 4 * i], 1e-9)) for i in range(n)]
    return dict(sigma_px=float(r.x[0]), widths_px=widths, contrast=float(np.median(amps)), rmse=float(np.sqrt(np.mean(r.fun ** 2))), n=n,
                at_bound=bool(r.x[0] <= 0.31 or r.x[0] >= 14.9))


def _fwhm_edges(d, p, frac=0.5):
    from .measure import _crossing
    lev = frac * d[p]
    return _crossing(d, p, lev, -1), _crossing(d, p, lev, +1)


def measure_loop(flat, ax, ay, rows=(15, 45), win=24, apex_win=(-18, 25), band=2):
    """Measure one loop around its apex (ax, ay) in PIXELS on a flat-fielded image (loops pointing up).

    Limbs: the profile averaged over `rows` below the apex; the nearest dark dip left of the apex and the
    nearest right of it are this loop's two limbs (not a neighbour's). Widths are half-depth (FWHM) widths;
    total = outer edge to outer edge. Apical: vertical half-depth thickness of the arc in a short column
    band through the apex. Returns dict (NaN where a part is not resolved) and the two limb sub-profiles."""
    from scipy.signal import find_peaks
    H, W = flat.shape
    out = dict(apex_x=float(ax), apex_y=float(ay), limb_left=np.nan, limb_right=np.nan, total=np.nan, apical=np.nan,
               depth_frac=np.nan)
    prof_pair = []
    r0, r1 = int(ay + rows[0]), min(H, int(ay + rows[1]))
    x0, x1 = max(0, int(ax - win)), min(W, int(ax + win) + 1)
    if r1 - r0 >= 8 and x1 - x0 >= 20:
        prof = flat[r0:r1, x0:x1].mean(0)
        bgw = flat[r0:r1, max(0, int(ax - 2 * win)):min(W, int(ax + 2 * win) + 1)].mean(0)
        bg = np.percentile(bgw, 85)
        d = bg - prof
        noise = np.median(np.abs(np.diff(prof))) / 0.6745 / np.sqrt(2)
        pk, pr = find_peaks(d, prominence=max(4 * noise, 0.004))
        rel = ax - x0
        left = [p for p in pk if p <= rel and rel - p <= win * 0.8]
        right = [p for p in pk if p > rel and p - rel <= win * 0.8]
        if left and right:
            pl, prr = max(left), min(right)
            if d[pl:prr + 1].min() < 0.5 * min(d[pl], d[prr]):           # the two limbs are resolved
                l0, l1 = _fwhm_edges(d, pl)
                q0, q1 = _fwhm_edges(d, prr)
                out.update(limb_left=float(l1 - l0), limb_right=float(q1 - q0), total=float(q1 - l0),
                           depth_frac=float(0.5 * (d[pl] + d[prr]) / bg))
                for p in (pl, prr):
                    a, b = max(0, p - 10), min(len(prof), p + 11)
                    prof_pair.append((prof[a:b] / bg, float(p - a)))
    c0, c1 = max(0, int(ay + apex_win[0])), min(H, int(ay + apex_win[1]))
    xa, xb = max(0, int(ax) - band), min(W, int(ax) + band + 1)
    if c1 - c0 > 10:
        col = flat[c0:c1, xa:xb].mean(1)
        dcol = np.percentile(flat[max(0, c0 - 20):c1, xa:xb].mean(1), 90) - col
        p = int(np.argmax(dcol))
        if 0 < p < len(dcol) - 1 and dcol[p] > 0:
            u0, u1 = _fwhm_edges(dcol, p)
            out["apical"] = float(u1 - u0)
    return out, prof_pair


def illumination_strength(green, mask=None, smooth_px=40):
    """Peak-to-peak relative brightness change of a fitted plane across the frame (lighting gradient)."""
    g = np.asarray(green, float)
    big = ndi.gaussian_filter(g, smooth_px)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    ok = np.ones_like(g, bool) if mask is None else ~ndi.binary_dilation(mask, iterations=smooth_px // 2)
    X = np.c_[np.ones(ok.sum()), xx[ok], yy[ok]]
    c, *_ = np.linalg.lstsq(X, big[ok], rcond=None)
    corners = [c[0] + c[1] * x + c[2] * y for x in (0, g.shape[1] - 1) for y in (0, g.shape[0] - 1)]
    return float((max(corners) - min(corners)) / big[ok].mean()), float(np.degrees(np.arctan2(c[2], c[1])))


def icc_oneway_unbalanced(groups):
    """ICC(1) for unequal group sizes (one-way random effects; n0 correction, Shrout and Fleiss / Donner)."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 1]
    k = len(groups)
    N = sum(len(g) for g in groups)
    if k < 2 or N - k < 1:
        return float("nan")
    gm = np.concatenate(groups).mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def texture_strength(rgb, um_per_px, corr_um=25.0, seed=0, dark_thresh=0.025):
    """Skin-texture amplitude in the units of optics.forward(texture_amp=...), measured on a real frame.

    Band-pass between corr/2 and 3*corr of the flat-fielded green channel, robust MAD, on background only:
    glare and anything darker than the local background by more than `dark_thresh` at limb scale (the
    capillaries) is masked and dilated. The SAME estimator, with the same masking rule, is applied to a
    synthetic texture field of amplitude 1 built like the forward model's; the ratio is the amplitude."""
    g = clean_green(rgb, glare_mask(rgb)).astype(float)
    glare = ndi.binary_dilation(glare_mask(rgb), iterations=8)
    c = corr_um / um_per_px

    def est(img, extra):
        rel = img / np.maximum(ndi.gaussian_filter(img, 4 * c), 1e-6) - 1.0
        fine = ndi.gaussian_filter(rel, 1.0) - ndi.gaussian_filter(rel, 6.0)
        dark = ndi.binary_dilation(fine < -dark_thresh, iterations=6)
        b = ndi.gaussian_filter(rel, c / 2) - ndi.gaussian_filter(rel, 3 * c)
        keep = ~(dark | extra)
        border = int(3 * c)
        keep[:border] = keep[-border:] = False
        keep[:, :border] = keep[:, -border:] = False
        v = b[keep] if keep.sum() > 500 else b.ravel()
        return float(1.4826 * np.median(np.abs(v - np.median(v)))), float(keep.mean())

    rng = np.random.default_rng(seed)
    syn = ndi.gaussian_filter(rng.normal(size=g.shape), c, mode="wrap")
    syn = (syn - syn.mean()) / syn.std()
    k, _ = est(1.0 + 0.01 * syn, np.zeros(g.shape, bool))
    v, frac = est(g, glare)
    return float(0.01 * v / k)
