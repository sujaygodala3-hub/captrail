"""Run the validated estimators on REAL captures.

Same code path as the simulations: flat-field, find the loop apexes of the distal row, then measure
apical width, limb widths and total loop width around each apex and report loops per mm.

Orientation: estimators assume loop apexes point UP (row 0). Tell the analyzer where the apexes point
in the saved image with apex='up'|'down'|'left'|'right', or use 'auto' (a heuristic: hairpin tops are
single rounded peaks, their open ends are paired; it reports how confident it is, verify by eye).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

from . import measure as M

ROT = {"up": 0, "left": -1, "down": 2, "right": 1}      # np.rot90 k that brings the apex direction to 'up'


def orient(img, apex="up"):
    """Rotate so loop apexes point up. 'left' means apexes point to the left in the image, and so on."""
    if apex not in ROT:
        raise ValueError(f"apex must be one of {list(ROT)} or 'auto'")
    return np.rot90(img, ROT[apex])


def find_apices(img, pixel_um, min_prominence_um=10.0, min_sep_um=20.0, min_area_um2=400.0,
                min_height_um=40.0, k_mad=3.0):
    """Apex positions (x_um, y_um from top) of loop-like dark structures, plus the working top-edge profile.

    Mirrors measure.count_apices, but returns where the apexes are so each loop can be measured."""
    f = ndi.gaussian_filter(M.flatten(img, pixel_um, fast=True), 1.0 / pixel_um)
    med = np.median(f)
    t = med - k_mad * 1.4826 * np.median(np.abs(f - med))
    dark = f < t
    lab, n = ndi.label(dark)
    if n == 0:
        return np.empty((0, 2)), f
    areas = ndi.sum(dark, lab, index=np.arange(1, n + 1))
    heights = np.array([(sl[0].stop - sl[0].start) * pixel_um for sl in ndi.find_objects(lab)])
    good = (areas * pixel_um ** 2 >= min_area_um2) & (heights >= min_height_um)
    keep = np.isin(lab, 1 + np.where(good)[0])
    ny = keep.shape[0]
    top = np.where(keep.any(0), keep.argmax(0), ny).astype(float)
    pk, _ = find_peaks(-top * pixel_um, prominence=min_prominence_um, distance=max(1, int(min_sep_um / pixel_um)))
    return np.c_[(pk + 0.5) * pixel_um, (top[pk] + 0.5) * pixel_um], f


def auto_orientation(img, pixel_um):
    """Guess which way the loop apexes point. Returns (direction, confidence, counts).

    Counts apexes in all four orientations. The axis along which loops are lined up shows many apexes;
    along it the true apex direction gives about N and the flipped one about 2N (paired limb ends), so
    the direction with the lower count wins and confidence = 1 - low/high (about 0.5 for a clean row;
    below 0.25 means 'look at the picture and set --apex yourself')."""
    counts = {name: len(find_apices(np.rot90(img, k), pixel_um)[0]) for name, k in ROT.items()}
    pairs = (("up", "down"), ("left", "right"))
    pair = max(pairs, key=lambda p: max(counts[p[0]], counts[p[1]]))
    lo, hi = sorted(pair, key=lambda n: counts[n])
    conf = 1.0 - counts[lo] / counts[hi] if counts[hi] > 0 else 0.0
    return lo, float(max(0.0, min(1.0, conf))), counts


def analyze_image(img, pixel_um, apex="up", half_width_um=45.0, limb_rows_um=(30.0, 80.0), margin_um=20.0):
    """Measure one capture. Returns (image_summary dict, loops DataFrame).

    img: 2D float array in DN or 0..1 (colour must already be converted). pixel_um: calibrated um per pixel.
    Loops whose apex is closer than `margin_um` to an image edge are skipped as incomplete.
    """
    info = {}
    if apex == "auto":
        apex, conf, counts = auto_orientation(img, pixel_um)
        info.update(orientation=apex, orientation_confidence=conf, orientation_counts=counts)
    else:
        info.update(orientation=apex)
    im = np.asarray(orient(img, apex), float)
    flat = M.flatten(im, pixel_um, fast=True)
    gate = M.quality_gate(im)
    apices, _ = find_apices(im, pixel_um)
    H, W = im.shape
    rows = []
    for x_um, y_um in apices:
        if x_um < margin_um or x_um > W * pixel_um - margin_um or y_um < margin_um:
            continue
        row = dict(apex_x_um=float(x_um), apex_y_um=float(y_um))
        row["apical_um"] = float(M.apical_width_fwhm(flat, pixel_um, x_um, flatfield=False))
        x0 = max(0, int((x_um - half_width_um) / pixel_um))
        x1 = min(W, int((x_um + half_width_um) / pixel_um) + 1)
        r0 = int((y_um + limb_rows_um[0]) / pixel_um)
        r1 = int((y_um + limb_rows_um[1]) / pixel_um)
        if r1 <= H and r1 > r0 and x1 - x0 > 8:
            sub = flat[:, x0:x1]
            lw = M.limb_widths_fwhm(sub, pixel_um, (r0, r1), flatfield=False)
            row["limb_left_um"], row["limb_right_um"] = (float(lw[0]), float(lw[1])) if len(lw) == 2 else (np.nan, np.nan)
            row["total_um"] = float(M.total_width_fwhm(sub, pixel_um, (r0, r1), flatfield=False))
        else:
            row.update(limb_left_um=np.nan, limb_right_um=np.nan, total_um=np.nan)
        rows.append(row)
    loops = pd.DataFrame(rows)
    width_mm = W * pixel_um / 1000.0
    info.update(n_apices=int(len(apices)), n_measured=int(len(loops)), width_mm=width_mm,
                density_per_mm=float(len(apices) / width_mm), **{f"qc_{k}": v for k, v in gate.items()})
    return info, loops
