"""Synthetic nailfold capillary geometry with exact ground truth.

Units: micrometers (um). Coordinates: x right, y up. A loop is a hairpin:
an arterial limb going up, a semicircular apex, a venous limb coming down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoopSpec:
    cx: float = 0.0            # x of loop centre
    y_base: float = 0.0        # y where the limbs start
    limb_gap: float = 40.0     # centre-to-centre distance between limbs
    limb_len: float = 120.0    # straight limb length below the apex
    width_art: float = 12.0    # arterial limb lumen width
    width_ven: float = 12.0    # venous limb lumen width
    apex_bulge: float = 0.0    # extra width at the apex (enlarged-loop model)
    wobble_amp: float = 0.0    # lateral sinusoid amplitude on the limbs (tortuosity)
    wobble_period: float = 60.0
    cross_y: float = -1.0      # >= 0: limbs cross once at this height above y_base (dataset class "crossing")
    cross_len: float = 30.0    # vertical extent of the crossing


def centerline(spec: LoopSpec, ds: float = 0.25):
    """Return (points[N,2], width[N], s[N]) for one loop, local coordinates."""
    r = spec.limb_gap / 2.0
    n_l = max(2, int(spec.limb_len / ds))
    n_a = max(2, int(np.pi * r / ds))
    y = np.linspace(0.0, spec.limb_len, n_l)
    wob = spec.wobble_amp * np.sin(2 * np.pi * y / spec.wobble_period) * np.sin(np.pi * y / spec.limb_len)
    crossed = spec.cross_y >= 0
    if crossed:      # smooth swap of the two limbs' sides across a window: art -r -> +r, ven +r -> -r
        u = np.clip((y - (spec.cross_y - spec.cross_len / 2)) / max(spec.cross_len, 1e-6), 0.0, 1.0)
        phi = np.pi * u * u * (3 - 2 * u)
        ax, vx = -r * np.cos(phi), r * np.cos(phi)
    else:
        ax, vx = np.full_like(y, -r), np.full_like(y, r)
    art = np.stack([ax + wob, y], 1)
    th = np.linspace(0.0, np.pi, n_a) if crossed else np.linspace(np.pi, 0.0, n_a)
    apex = np.stack([r * np.cos(th), spec.limb_len + r * np.sin(th)], 1)
    ven = np.stack([vx[::-1] + wob[::-1], y[::-1]], 1)
    pts = np.concatenate([art, apex[1:], ven[1:]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = s / s[-1]
    w = spec.width_art + (spec.width_ven - spec.width_art) * t
    if spec.apex_bulge:
        s_apex = s[n_l - 1] + 0.5 * np.pi * r
        sig = max(r, 10.0)
        w = w + spec.apex_bulge * np.exp(-0.5 * ((s - s_apex) / sig) ** 2)
    return pts, w, s


def centerline_length(spec: LoopSpec) -> float:
    return float(centerline(spec)[2][-1])


def tortuosity_index(spec: LoopSpec) -> float:
    """Path length over the loop's own straight-hairpin length (1.0 = no wobble).

    A plain path/chord ratio is meaningless for a hairpin, so tortuosity is
    measured against the un-wobbled reference loop with the same gap and length.
    """
    ref = LoopSpec(limb_gap=spec.limb_gap, limb_len=spec.limb_len)
    return centerline_length(spec) / centerline_length(ref)


def render_mask(loops, fov_um, res_um):
    """Rasterize loops to a boolean lumen mask (row 0 = top of the field).

    The lumen is the union of discs of radius width/2 stamped along each centerline, so
    variable width (apical bulge) is exact even where the bulge is wider than the loop radius.
    """
    W, H = fov_um
    nx, ny = int(round(W / res_um)), int(round(H / res_um))
    mask = np.zeros((ny, nx), bool)
    for sp in loops:
        pts, w, _ = centerline(sp, ds=res_um)
        pts = pts + np.array([sp.cx, sp.y_base])
        for (px, py), wi in zip(pts, w):
            r = wi / 2.0
            ix0, ix1 = max(0, int((px - r) / res_um) - 1), min(nx, int(np.ceil((px + r) / res_um)) + 2)
            iy0, iy1 = max(0, int((H - py - r) / res_um) - 1), min(ny, int(np.ceil((H - py + r) / res_um)) + 2)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            X = (np.arange(ix0, ix1) + 0.5) * res_um
            Y = H - (np.arange(iy0, iy1) + 0.5) * res_um
            mask[iy0:iy1, ix0:ix1] |= ((X[None, :] - px) ** 2 + (Y[:, None] - py) ** 2) <= r * r
    return mask


def loop_array(n, spacing_um, rng, jitter=0.15, **kw):
    """n loops in a row (a distal-row model) with jittered spacing. Returns specs, field width."""
    specs, x = [], spacing_um * 0.6
    for _ in range(n):
        x += spacing_um * (1 + jitter * rng.uniform(-1, 1))
        specs.append(LoopSpec(cx=x, y_base=10.0, **kw))
    return specs, x + spacing_um * 0.6


def make_loop(limb_um, apical_um, total_um, limb_len=120.0, **kw):
    """Loop with separate limb, apical and total widths (the three EULAR-relevant sizes).

    total_um is outer edge to outer edge across the two limbs, so the limb centre gap is
    total_um - limb_um. apical_um is the lumen thickness at the apex (>= limb_um).
    """
    bulge = max(0.0, apical_um - limb_um)
    return LoopSpec(limb_gap=total_um - limb_um, limb_len=limb_len, width_art=limb_um,
                    width_ven=limb_um, apex_bulge=bulge, **kw)


def true_apical_width(spec: LoopSpec) -> float:
    _, w, s = centerline(spec)
    r = spec.limb_gap / 2.0
    n_l = max(2, int(spec.limb_len / 0.25))
    return float(w[n_l - 1: n_l - 1 + int(np.pi * r / 0.25)].max())


def true_total_width(spec: LoopSpec) -> float:
    return float(spec.limb_gap + 0.5 * (spec.width_art + spec.width_ven))


APICAL_WIDTHS = [16, 20, 25, 30, 35, 40, 45, 50, 55, 60]     # dense around the 50 um giant threshold
LIMB_WIDTHS = [8, 10, 12, 16, 20]
TOTAL_WIDTHS = [30, 35, 40, 45]


def loop_row(n, spacing_um, rng, jitter=0.15, **kw):
    """n loops filling a field exactly n*spacing wide (true density = 1000/spacing per mm).

    Loops sit at (i + 0.5) * spacing plus jitter. Returns (specs, field_width_um).
    """
    specs = []
    for i in range(n):
        x = (i + 0.5) * spacing_um + jitter * spacing_um * rng.uniform(-1, 1)
        specs.append(LoopSpec(cx=x, y_base=10.0, **kw))
    return specs, n * spacing_um
