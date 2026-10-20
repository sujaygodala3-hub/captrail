"""Automated measurements on detector images.

Width estimators work on a region of interest (ROI) covering the straight
parts of two vertical limbs. Real images need orientation handling first; this
is the controlled-test version used to map what the optics allow.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu


def flatten(img, pixel_um, max_width_um=60.0, fast=False):
    """Divide out slow illumination: closing removes dark vessels, then smooth.

    fast=True estimates the background on a block-averaged copy (about 4 um per block) and
    interpolates it back: same result for slow lighting, much quicker on large fields."""
    if fast and pixel_um < 3.0:
        k = int(round(4.0 / pixel_um))
        ny, nx = img.shape
        small = img[: ny // k * k, : nx // k * k].reshape(ny // k, k, nx // k, k).mean((1, 3))
        bg = flatten_bg(small, pixel_um * k, max_width_um)
        bg = ndi.zoom(bg, k, order=1, mode="nearest")
        bg = np.pad(bg, ((0, max(0, ny - bg.shape[0])), (0, max(0, nx - bg.shape[1]))), mode="edge")[:ny, :nx]
        return img / np.maximum(bg, 1e-3)
    size = int(round(1.5 * max_width_um / pixel_um)) | 1
    sm = ndi.gaussian_filter(img, 1.5 / pixel_um)
    bg = ndi.gaussian_filter(ndi.grey_closing(sm, size=size), size / 4.0)
    return img / np.maximum(bg, 1e-3)


def flatten_bg(img, pixel_um, max_width_um=60.0):
    size = int(round(1.5 * max_width_um / pixel_um)) | 1
    sm = ndi.gaussian_filter(img, 1.5 / pixel_um)
    return ndi.gaussian_filter(ndi.grey_closing(sm, size=size), size / 4.0)


def _crossing(d, p, level, step, rise_tol=0.10):
    """Sub-sample position where d falls to `level` walking from peak p. Returns NaN if the walk runs
    off the profile OR meets the neighbouring structure (d rises again by more than rise_tol*d[p]
    before reaching `level`: the valley between two limbs stays above half depth, so the limbs are unresolved)."""
    i = p
    runmin = d[p]
    noise = np.median(np.abs(np.diff(d))) / 0.6745 / np.sqrt(2)
    tol = max(rise_tol * d[p], 3.0 * noise)      # a real neighbouring limb rises by far more than noise
    while 0 <= i + step < len(d) and d[i + step] > level:
        i += step
        runmin = min(runmin, d[i])
        if runmin < level + 0.3 * (d[p] - level) and d[i] > runmin + tol:   # plateau noise near the top is ignored
            return np.nan
    j = i + step
    if not (0 <= j < len(d)):
        return np.nan
    return i + (d[i] - level) / (d[i] - d[j]) * step


def limb_widths_fwhm(img, pixel_um, roi_rows, frac=0.5, flatfield=True):
    """Width of the two deepest dips in the row-averaged profile, at `frac` of dip depth.

    Returns array of 2 widths in um (NaN if fewer than two limbs resolved).
    """
    f = flatten(img, pixel_um) if flatfield else img
    prof = f[roi_rows[0]:roi_rows[1]].mean(0)
    d = np.percentile(prof, 90) - prof   # baseline = bright background (vessels can cover >50% of a profile)
    noise = np.median(np.abs(np.diff(prof))) / 0.6745 / np.sqrt(2)
    peaks, props = find_peaks(d, prominence=max(6 * noise, 0.01))
    if len(peaks) < 2:
        return np.array([np.nan, np.nan])
    top = peaks[np.argsort(props["prominences"])[-2:]]
    a, b = sorted(top)
    if d[a:b + 1].min() > frac * min(d[a], d[b]):      # valley between the limbs stays above the crossing level: unresolved
        return np.array([np.nan, np.nan])
    out = []
    for p in sorted(top):
        lev = frac * d[p]
        xl, xr = _crossing(d, p, lev, -1), _crossing(d, p, lev, +1)
        out.append((xr - xl) * pixel_um)    # NaN when the limbs are unresolved
    return np.array(out)


def limb_widths_otsu(img, pixel_um, roi_rows, flatfield=True):
    """Naive area estimator: Otsu on the whole image, dark area in ROI / (rows * 2 limbs)."""
    f = flatten(img, pixel_um) if flatfield else img
    t = threshold_otsu(ndi.gaussian_filter(f, 0.7 / pixel_um))
    dark = f[roi_rows[0]:roi_rows[1]] < t
    w = dark.sum() / dark.shape[0] / 2.0 * pixel_um
    return np.array([w, w])


def count_loops(img, pixel_um, min_area_um2=150.0):
    """Count connected dark objects after flat-field + Otsu. One loop = one object."""
    f = ndi.gaussian_filter(flatten(img, pixel_um), 1.0 / pixel_um)
    dark = f < threshold_otsu(f)
    lab, n = ndi.label(dark)
    if n == 0:
        return 0
    areas = ndi.sum(dark, lab, index=np.arange(1, n + 1))
    return int((areas * pixel_um ** 2 >= min_area_um2).sum())


# ---------------------------------------------------------------------------
# S1/S2 additions: fixed-threshold, apical and total width, apex-based density, quality gate
# ---------------------------------------------------------------------------
def _runs_below(prof, thr):
    lab, n = ndi.label(prof < thr)
    return [np.where(lab == i)[0] for i in range(1, n + 1)]


def limb_widths_fixed(img, pixel_um, roi_rows, thr=0.90, flatfield=True):
    """Fixed-threshold estimator: width of the two deepest runs below `thr` x background.

    Unlike half-depth this does not adapt to contrast, so it is biased when contrast is low.
    Returns 2 widths in um (NaN if fewer than two runs).
    """
    f = flatten(img, pixel_um) if flatfield else img
    prof = f[roi_rows[0]:roi_rows[1]].mean(0)
    prof = prof / np.percentile(prof, 90)
    runs = _runs_below(prof, thr)
    if len(runs) < 2:
        return np.array([np.nan, np.nan])
    runs = sorted(runs, key=lambda r: prof[r].min())[:2]
    out = []
    for r in sorted(runs, key=lambda r: r[0]):
        i0, i1 = r[0], r[-1]
        left = i0 - (thr - prof[i0]) / max(prof[i0 - 1] - prof[i0], 1e-9) if i0 > 0 else i0
        right = i1 + (thr - prof[i1]) / max(prof[i1 + 1] - prof[i1], 1e-9) if i1 < len(prof) - 1 else i1
        out.append((right - left) * pixel_um)
    return np.array(out)


def apical_width_fwhm(img, pixel_um, cx_um, half_band_um=3.0, flatfield=True):
    """Apical width: vertical half-depth width of the dark arc through the loop's centre column.

    cx_um is the loop centre x. The apex is the highest dark structure in that column band.
    """
    f = flatten(img, pixel_um) if flatfield else img
    i = int(np.floor(cx_um / pixel_um))      # column i is centred at (i + 0.5) * pixel_um
    h = max(1, int(round(half_band_um / pixel_um)))
    col = f[:, max(0, i - h): i + h + 1].mean(1)
    d = np.percentile(col, 90) - col
    p = int(np.argmax(d))
    lev = 0.5 * d[p]
    up, dn = _crossing(d, p, lev, -1), _crossing(d, p, lev, +1)
    return (dn - up) * pixel_um


def total_width_fwhm(img, pixel_um, roi_rows, flatfield=True):
    """Total loop width: outer half-depth edge of the left limb to outer edge of the right limb."""
    f = flatten(img, pixel_um) if flatfield else img
    prof = f[roi_rows[0]:roi_rows[1]].mean(0)
    d = np.percentile(prof, 90) - prof
    lev = 0.5 * d.max()
    above = np.where(d > lev)[0]
    if len(above) < 2:
        return float("nan")
    i0, i1 = above[0], above[-1]
    left = i0 - (d[i0] - lev) / max(d[i0] - d[i0 - 1], 1e-9) if i0 > 0 else i0
    right = i1 + (d[i1] - lev) / max(d[i1] - d[i1 + 1], 1e-9) if i1 < len(d) - 1 else i1
    return (right - left) * pixel_um


def count_apices(img, pixel_um, min_prominence_um=10.0, min_sep_um=20.0, min_area_um2=400.0, min_height_um=40.0, k_mad=3.0, threshold="mad"):
    """Apex-based loop count for a distal row: local highs of the top edge of the dark mask.

    One loop = one apex, so two merged loops still count as two if their apexes stay apart,
    which is what density estimation needs. Returns the number of apexes.
    """
    f = ndi.gaussian_filter(flatten(img, pixel_um, fast=True), 1.0 / pixel_um)
    if threshold == "otsu":
        t = threshold_otsu(f)
    else:   # background level and spread from the median and MAD, so texture and noise set the floor
        med = np.median(f)
        t = med - k_mad * 1.4826 * np.median(np.abs(f - med))
    dark = f < t
    lab, n = ndi.label(dark)
    if n == 0:
        return 0
    areas = ndi.sum(dark, lab, index=np.arange(1, n + 1))
    heights = np.array([(sl[0].stop - sl[0].start) * pixel_um for sl in ndi.find_objects(lab)])
    good = (areas * pixel_um ** 2 >= min_area_um2) & (heights >= min_height_um)   # loops are long; texture blobs are not
    keep = np.isin(lab, 1 + np.where(good)[0])
    ny = keep.shape[0]
    top = np.where(keep.any(0), keep.argmax(0), ny).astype(float) * pixel_um
    pad = np.full(1, ny * pixel_um)          # background level at both ends so an apex at the image edge can be a peak
    pk, _ = find_peaks(-np.r_[pad, top, pad], prominence=min_prominence_um, distance=max(1, int(min_sep_um / pixel_um)))
    return int(len(pk))


def density_per_mm(img, pixel_um, **kw):
    width_mm = img.shape[1] * pixel_um / 1000.0
    return count_apices(img, pixel_um, **kw) / width_mm


def quality_gate(img, bits=8, min_focus=None, max_sat_frac=0.02, mean_range=(0.15, 0.85)):
    """Image-quality gate (guide section 'Quality gate'): focus, clipping, exposure.

    Focus is the variance of the Laplacian, compared with `min_focus` if given (set it from
    your own known-good captures; it depends on resolution). Returns dict with `pass`.
    """
    q = 2 ** bits - 1
    x = np.asarray(img, float)
    x = x / q if x.max() > 1.5 else x
    lap = ndi.laplace(x)
    focus = float(lap.var())
    sat = float(((x >= 0.995) | (x <= 0.005)).mean())
    mean = float(x.mean())
    ok = sat <= max_sat_frac and mean_range[0] <= mean <= mean_range[1] and (min_focus is None or focus >= min_focus)
    return dict(focus=focus, saturated_frac=sat, mean=mean, **{"pass": bool(ok)})
