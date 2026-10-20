"""Fit the simulator's Look ranges to real frames by matching image statistics (domain_gap features).

Coordinate search over a few interpretable knobs (contrast, blur, noise, colour cast, shading...), with
common random numbers so the objective is smooth. Objective: mean squared standardized mean difference over
the feature set (not the classifier AUC, which saturates at 1.0 and gives no gradient). Report the AUC after.

Warning on small real sets: with a few dozen real images this fits the SAMPLE, not the camera. Use the
full Morphology_detection images (about 1.3k) and keep a held-out half to report the final AUC.
"""
from __future__ import annotations

import copy
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from . import domain_gap as DG, render_rgb as R

SCALE_KNOBS = {"contrast": ("contrast", 0.03, 1.0), "sigma": ("sigma_um", 0.3, 12.0), "tint": ("tint_amp", 0.0, 0.5),
               "illum": ("illum", 0.0, 0.8), "texture": ("texture", 0.0, 0.25), "sharpen": ("sharpen", 0.0, 1.5),
               "read_noise": ("read_noise", 0.001, 0.06)}


def apply(look, theta):
    kw = {}
    for k, s in theta.items():
        if k in SCALE_KNOBS:
            fld, lo, hi = SCALE_KNOBS[k]
            kw[fld] = tuple(float(np.clip(v * s, lo, hi)) for v in getattr(look, fld))
        elif k == "photons":                     # more photons = less noise
            kw["photons"] = tuple(float(np.clip(v * s, 50, 5e4)) for v in look.photons)
        elif k == "sat":
            kw["sat"] = float(np.clip(look.sat * s, 0.1, 1.6))
        elif k == "bottom":
            kw["bottom_shift"] = tuple(float(np.clip(v + s, -0.6, 0.4)) for v in look.bottom_shift)
    return replace(look, **kw)


def _render(args):
    look, seed, size = args
    return DG.features(R.render_scene(np.random.default_rng(seed), look)["img"], size)


def synth_feats(look, n, seed, size, pool=None):
    jobs = [(look, seed + i, size) for i in range(n)]
    return np.stack(pool.map(_render, jobs) if pool else [_render(j) for j in jobs])


def objective(real_f, synth_f):
    pooled = np.sqrt(0.5 * (real_f.var(0) + synth_f.var(0))) + 1e-9
    smd = (synth_f.mean(0) - real_f.mean(0)) / pooled
    return float(np.mean(np.minimum(smd ** 2, 9.0)))


def fit_look(real_imgs, base=None, rounds=3, n_synth=30, seed=1000, size=(224, 168), workers=4, log=print):
    base = base or R.Look.dataset()
    real_f = DG.feature_matrix(real_imgs, size)
    theta = {k: 1.0 for k in list(SCALE_KNOBS) + ["photons", "sat"]}
    theta["bottom"] = 0.0
    hist = []
    with Pool(workers) as pool:
        look = apply(base, theta)
        best = objective(real_f, synth_feats(look, n_synth, seed, size, pool))
        hist.append(("start", best))
        for r in range(rounds):
            for k in list(theta):
                for step in ((0.7, 1.4) if k != "bottom" else (-0.1, 0.1)):
                    cand = dict(theta)
                    cand[k] = theta[k] * step if k != "bottom" else theta[k] + step
                    sc = objective(real_f, synth_feats(apply(base, cand), n_synth, seed, size, pool))
                    if sc < best - 1e-4:
                        best, theta = sc, cand
                        hist.append((f"{k}={cand[k]:.3g}", best))
                        log(f"round {r} {k:<10} -> objective {best:.3f}")
        look = apply(base, theta)
    return look, dict(theta=theta, history=hist, objective=best)
