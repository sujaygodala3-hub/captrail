"""The four capillary morphology classes of the Capillary Dataset (Nguyen and Jeong, KDD 2025):
bushy, crossing, hairpin, tortuous. Parametric proxies built from the exact-truth loop geometry.

Class names and ORDER match the dataset's data.yaml (YOLO class ids 0..3). Class frequencies
(PRIOR) were read off the label-count bar chart in the dataset repo's training run (about 550,
3700, 5800, 1130 boxes): approximate, replace them with counts from `dataset_profile.py` once the
Morphology_detection folder is local. Shapes are PROXIES: judge them with `domain_gap.py`, not by eye.
"""
from __future__ import annotations

import numpy as np

from . import geometry as G

CLASSES = ["bushy", "crossing", "hairpin", "tortuous"]
PRIOR = {"bushy": 550.0, "crossing": 3700.0, "hairpin": 5800.0, "tortuous": 1130.0}


def class_probs(counts=None, balance=0.0):
    """Class sampling probabilities. balance 0 = dataset frequency, 1 = uniform (for rare classes)."""
    c = np.array([(counts or PRIOR)[k] for k in CLASSES], float)
    p = c / c.sum()
    p = (1 - balance) * p + balance / len(CLASSES)
    return p / p.sum()


def _limb_loop(rng, **kw):
    limb = rng.uniform(6.5, 12)
    total = rng.uniform(26, 50)
    apical = limb + rng.choice([0.0, rng.uniform(0, 14)])
    spec = G.make_loop(limb, apical, total, limb_len=rng.uniform(70, 160), **kw)
    return spec, dict(limb_um=float(limb), apical_um=float(G.true_apical_width(spec)), total_um=float(G.true_total_width(spec)))


def sample_object(cls, rng):
    """Return (list[LoopSpec], truth dict) for one object of class `cls`. Coordinates local: cx, y_base set later."""
    if cls == "hairpin":
        sp, truth = _limb_loop(rng, wobble_amp=rng.uniform(0, 1.5), wobble_period=rng.uniform(50, 90))
        return [sp], truth
    if cls == "tortuous":
        sp, truth = _limb_loop(rng, wobble_amp=rng.uniform(4, 9), wobble_period=rng.uniform(30, 65))
        sp.limb_len = max(sp.limb_len, 90.0)
        return [sp], truth
    if cls == "crossing":
        sp, truth = _limb_loop(rng, wobble_amp=rng.uniform(0, 2.5), wobble_period=rng.uniform(40, 80))
        sp.limb_len = max(sp.limb_len, 90.0)
        sp.limb_gap = min(sp.limb_gap, 30.0)
        sp.cross_len = rng.uniform(18, 40)
        sp.cross_y = rng.uniform(0.3, 0.65) * sp.limb_len
        return [sp], dict(truth, apical_um=float(G.true_apical_width(sp)), total_um=float(G.true_total_width(sp)))
    if cls == "bushy":      # a tangle of 3 to 4 short, curly sub-loops
        specs = []
        for i in range(int(rng.integers(3, 5))):
            limb = rng.uniform(7, 11)
            sp = G.make_loop(limb, limb + rng.uniform(0, 5), rng.uniform(18, 28), limb_len=rng.uniform(40, 80),
                             wobble_amp=rng.uniform(3, 6), wobble_period=rng.uniform(25, 45))
            sp.cx = rng.uniform(-14, 14)
            sp.y_base = rng.uniform(0, 25)         # offset from the object's base, added later
            specs.append(sp)
        return specs, dict(limb_um=None, apical_um=None, total_um=None)
    raise KeyError(cls)


def object_bbox(specs):
    """Axis-aligned bounding box (x0, y0, x1, y1) in um, y up, of the union of specs placed at their cx, y_base."""
    xs0, ys0, xs1, ys1 = [], [], [], []
    for sp in specs:
        pts, w, _ = G.centerline(sp, ds=1.0)
        pts = pts + np.array([sp.cx, sp.y_base])
        xs0.append((pts[:, 0] - w / 2).min()); xs1.append((pts[:, 0] + w / 2).max())
        ys0.append((pts[:, 1] - w / 2).min()); ys1.append((pts[:, 1] + w / 2).max())
    return min(xs0), min(ys0), max(xs1), max(ys1)
