"""Run the Capillary core experiments and write CSVs + figures to results/.

    python run_experiments.py            # full run (~2-4 min)
    python run_experiments.py --quick    # fewer replicates

E1 blur sweep      width bias vs PSF sigma, per true width, two estimators
E2 noise sweep     width error spread vs shot noise (photons/pixel)
E3 illumination    effect of a lighting gradient, with and without flat-fielding
E4 pixel size      detector pixel size vs width error
E5 loop counting   count error vs blur, per true density
E6 propagation     width error pushed through Poiseuille: R ~ r^-4, wall shear ~ r^-3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from capsim import geometry as G, measure as M, optics as O

RES = 0.5                      # fine grid, um/px
OUT = Path(__file__).parent / "results"
WIDTHS = [8, 12, 20, 40]       # um: small, normal, enlarged, giant-range
ROI_UM = (60, 140)             # straight-limb band, um from the bottom of the field
FOV = (160, 220)


def single_loop_mask(w, gap=60.0):
    sp = G.LoopSpec(cx=80, y_base=10, limb_gap=gap, limb_len=150, width_art=w, width_ven=w)
    return G.render_mask([sp], FOV, RES)


def roi_rows(pixel_um):
    top_um = FOV[1] - ROI_UM[1]
    return int(top_um / pixel_um), int((FOV[1] - ROI_UM[0]) / pixel_um)


def measure_both(img, px, flat=True):
    rr = roi_rows(px)
    return (M.limb_widths_fwhm(img, px, rr, flatfield=flat).mean(),
            M.limb_widths_otsu(img, px, rr, flatfield=flat).mean())


def e1_blur(reps):
    rows, sigmas = [], [0, 1, 2, 3, 4, 6, 8, 10]
    for w in WIDTHS:
        mask = single_loop_mask(w)
        for s in sigmas:
            for r in range(reps):
                rng = np.random.default_rng(1000 * w + 10 * s + r)
                img = O.forward(mask, RES, sigma_um=s, photons=4000, read_noise=0.005, rng=rng)
                f, o = measure_both(img, 1.0)
                rows.append(dict(width=w, sigma=s, rep=r, fwhm=f, otsu=o))
    return pd.DataFrame(rows)


def e2_noise(reps):
    rows = []
    for w in WIDTHS:
        mask = single_loop_mask(w)
        for ph in [200, 500, 1000, 2000, 5000, 20000]:
            for r in range(reps):
                rng = np.random.default_rng(7 * ph + r)
                img = O.forward(mask, RES, sigma_um=2.0, photons=ph, read_noise=0.005, rng=rng)
                f, o = measure_both(img, 1.0)
                rows.append(dict(width=w, photons=ph, rep=r, fwhm=f, otsu=o))
    return pd.DataFrame(rows)


def e3_illum(reps):
    rows, mask = [], single_loop_mask(12)
    for s in [0, 0.1, 0.2, 0.3, 0.5]:
        for flat in (True, False):
            for r in range(reps):
                rng = np.random.default_rng(31 * r + int(100 * s))
                img = O.forward(mask, RES, sigma_um=2.0, illum_strength=s, illum_angle_deg=20,
                                photons=4000, read_noise=0.005, rng=rng)
                f, o = measure_both(img, 1.0, flat=flat)
                rows.append(dict(illum=s, flatfield=flat, rep=r, fwhm=f, otsu=o))
    return pd.DataFrame(rows)


def e4_pixel(reps):
    rows = []
    for w in WIDTHS:
        mask = single_loop_mask(w)
        for px in [0.5, 1.0, 2.0, 3.0, 4.0]:
            for r in range(reps):
                rng = np.random.default_rng(13 * r + int(10 * px))
                img = O.forward(mask, RES, sigma_um=1.5, pixel_um=px, photons=4000, read_noise=0.005, rng=rng)
                f, o = measure_both(img, px)
                rows.append(dict(width=w, pixel=px, rep=r, fwhm=f, otsu=o))
    return pd.DataFrame(rows)


def e5_count(reps):
    """Count 7 loops (a normal ~7/mm distal row) at low-to-normal contrast, with noise.

    Real nailfold contrast is low, so contrast (not blur) is the realistic stressor here.
    """
    rows, n, spacing = [], 7, 1000.0 / 7
    for contrast in [0.05, 0.10, 0.20, 0.35]:
        for s in [0, 2, 4, 6, 8]:
            for r in range(reps):
                rng = np.random.default_rng(97 * r + 10 * s + int(100 * contrast))
                specs, W = G.loop_array(n, spacing, rng, limb_gap=40, limb_len=100, width_art=12, width_ven=12)
                mask = G.render_mask(specs, (W, 220), RES)
                img = O.forward(mask, RES, sigma_um=s, contrast=contrast, photons=1500, read_noise=0.01, rng=rng)
                rows.append(dict(contrast=contrast, sigma=s, rep=r, true=n, counted=M.count_loops(img, 1.0)))
    return pd.DataFrame(rows)


def e6_propagation(e1: pd.DataFrame):
    """Turn each measured width into a resistance / shear-stress error under Poiseuille flow.

    R ~ r^-4 (fixed length, viscosity); wall shear at fixed flow ~ r^-3. Idealized straight
    tube only; real capillary blood flow is not Poiseuille, so read this as an amplification
    argument, not a prediction.
    """
    d = e1.copy()
    rel = d["fwhm"] / d["width"]
    d["R_err_pct"] = ((rel ** -4) - 1) * 100
    d["WSS_err_pct"] = ((rel ** -3) - 1) * 100
    d["w_err_pct"] = (rel - 1) * 100
    return d


def main(quick=False):
    OUT.mkdir(exist_ok=True)
    reps = 6 if quick else 20
    e1 = e1_blur(reps); e1.to_csv(OUT / "e1_blur.csv", index=False)
    e2 = e2_noise(reps); e2.to_csv(OUT / "e2_noise.csv", index=False)
    e3 = e3_illum(reps); e3.to_csv(OUT / "e3_illumination.csv", index=False)
    e4 = e4_pixel(reps); e4.to_csv(OUT / "e4_pixel.csv", index=False)
    e5 = e5_count(max(4, reps // 2)); e5.to_csv(OUT / "e5_count.csv", index=False)
    e6 = e6_propagation(e1); e6.to_csv(OUT / "e6_propagation.csv", index=False)
    import plots
    plots.make_all(e1, e2, e3, e4, e5, e6, OUT)
    print("done ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
