"""Colour, Celestron-style frames with YOLO labels, built on the exact-truth loop geometry.

Forward model per colour channel: ideal_c = bg_c(y) * (1 - k_c * contrast * lumen) * (1 + texture) * lighting,
then depth-varying Gaussian blur, area-integrating sensor, Poisson + read noise, optional ISP sharpening
and JPEG. Hemoglobin absorbs green most and red least, so k = (0.35, 1.0, 0.8): loops come out red-brown.
Everything here is a MODEL with assumed ranges (Look): the ranges are meant to be re-fitted to real
Celestron frames and to the dataset (`dataset_profile.py`, `adapt.py`), then checked with `domain_gap.py`.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from . import geometry as G, morphology as MO

ABSORB = np.array([0.45, 1.0, 0.9])
BACKGROUNDS = {                       # display-space skin colours, RGB 0..1 (read off sample frames: pale grey-green, bluish, pink, salmon, olive)
    "grey_green": (0.62, 0.68, 0.62), "bluish": (0.58, 0.62, 0.72), "pink": (0.80, 0.62, 0.60),
    "salmon": (0.82, 0.45, 0.40), "olive": (0.62, 0.62, 0.45),
    "peach": (0.85, 0.60, 0.38), "orange": (0.86, 0.56, 0.32)}       # Celestron raw frames (Sujay session, 2026-10-21)
LOWER = {"grey_green": (0.78, 0.62, 0.60), "bluish": (0.74, 0.60, 0.62), "pink": (0.82, 0.58, 0.56),
         "salmon": (0.84, 0.40, 0.36), "olive": (0.72, 0.58, 0.50), "peach": (0.84, 0.54, 0.34), "orange": (0.80, 0.48, 0.26)}      # tone of the lower (proximal) skin: pinker
DATASET_PX_UM = 0.96      # INFERRED: median hairpin box is ~0.065 of image width; with a ~40 um loop that is ~620 um across 640 px. ASSUMPTION.


@dataclass
class Look:
    """Ranges the sampler draws from. Replace with fitted values from real frames."""
    size: tuple = (640, 480)                         # Celestron capture is 640x480
    px_um: float = DATASET_PX_UM
    sigma_um: tuple = (1.2, 6.0)                     # log-uniform PSF sigma
    sigma_ramp: tuple = (0.6, 1.6)                   # top vs bottom blur ratio (tilted nail plane)
    contrast: tuple = (0.15, 0.45)                   # lumen contrast in the green channel
    photons: tuple = (300.0, 3000.0)
    read_noise: tuple = (0.004, 0.02)
    illum: tuple = (0.0, 0.25)
    texture: tuple = (0.0, 0.08)
    bg_names: tuple = tuple(BACKGROUNDS)
    n_objects: tuple = (5, 16)
    row_y: tuple = (0.12, 0.55)                      # apex height of the loop row, fraction from image top
    row_slope: tuple = (-0.12, 0.12)
    bubbles: tuple = (0, 3)
    specks: tuple = (0, 6)
    sharpen: tuple = (0.0, 0.6)
    jpeg_q: tuple = (65, 95)
    fine_um: float = 1.0
    bg_override: tuple | None = None                 # fixed (r, g, b), for matching one real session
    sat: float = 1.0                                 # background saturation scale (1 = palette as listed)
    tint_amp: tuple = (0.0, 0.12)                    # large-scale per-channel colour drift (greenish, pinkish zones)
    bottom_shift: tuple = (-0.25, 0.05)              # relative brightness change from top to bottom of the frame
    glare: tuple = (0, 0)                            # large specular LED reflections (count range): flat cream ellipses, clipped
    limb_len_scale: float = 1.0                      # visible loop length multiplier (Celestron frames show long limbs)
    striation: tuple = (0.0, 0.0)                    # faint vertical streak texture along the loops (fractional amplitude)

    def save(self, path):
        import json
        from dataclasses import asdict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        import json
        d = json.loads(Path(path).read_text())
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    @classmethod
    def dataset(cls):
        """Looks like the published dataset frames: paler, less saturated, bottom darker and pinker."""
        return cls(sat=0.7, contrast=(0.12, 0.4), tint_amp=(0.0, 0.06), sigma_um=(0.8, 3.5), texture=(0.02, 0.10), illum=(0.1, 0.45), bottom_shift=(-0.45, -0.05), bg_names=("grey_green", "bluish", "pink", "olive"))

    @classmethod
    def celestron(cls):
        """Looks like the RAW Celestron scan-page frames measured in the 2026-10-21 session: peach/orange cast,
        large LED glare ellipses, blur sigma about 1.9 px, lumen contrast about 0.1, very low noise after the
        camera's processing. px_um 1.6 is an anatomical ESTIMATE until the ruler (S0) is measured."""
        return cls(px_um=1.6, sat=1.0, contrast=(0.06, 0.16), sigma_um=(2.2, 4.2), sigma_ramp=(0.8, 1.25),
                   bg_names=("peach", "orange"), bubbles=(0, 2), specks=(0, 3), glare=(0, 3), tint_amp=(0.0, 0.0),
                   photons=(2e4, 1e5), read_noise=(0.001, 0.004), illum=(0.08, 0.28), texture=(0.0, 0.01),
                   limb_len_scale=1.8, striation=(0.005, 0.02),
                   jpeg_q=(88, 98), sharpen=(0.0, 0.3), n_objects=(5, 10), bottom_shift=(-0.15, 0.05))


def _lu(rng, lo_hi):
    return float(np.exp(rng.uniform(math.log(lo_hi[0]), math.log(lo_hi[1]))))


def _u(rng, lo_hi):
    return float(rng.uniform(*lo_hi))


def sharpness(img_u8):
    """Variance of the Laplacian of the grey image (higher = sharper). Same family as a focus score;
    the scan page's own 0 to 100 'Sharpness' formula is unknown, so this is not on that scale."""
    import cv2
    g = cv2.cvtColor(np.asarray(img_u8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _place(rng, look, W, H, n, probs):
    objs, ys = [], []
    row = _u(rng, look.row_y)
    slope = _u(rng, look.row_slope)
    xs = np.sort(rng.uniform(0.04, 0.96, n)) * W
    for x in xs:
        cls = MO.CLASSES[int(rng.choice(len(MO.CLASSES), p=probs))]
        specs, truth = MO.sample_object(cls, rng)
        if look.limb_len_scale != 1.0:
            for sp in specs:
                sp.limb_len *= look.limb_len_scale
                if sp.cross_y >= 0:
                    sp.cross_y *= look.limb_len_scale
        top_frac = np.clip(row + slope * (x / W - 0.5) + rng.normal(0, 0.05), 0.05, 0.8)
        _, _, _, y1 = MO.object_bbox(specs)           # local top of the object (y up, relative to y_base 0)
        base = H * (1 - top_frac) - y1                # place so the apex sits at top_frac
        for sp in specs:
            sp.cx += x
            sp.y_base += base
        objs.append(dict(cls=cls, specs=specs, truth=truth))
    return objs


def render_scene(rng, look: Look | None = None, probs=None, n_objects=None, sigma_um=None):
    """One colour frame. Returns dict(img=uint8 HxWx3, boxes=[(cls_id, cx, cy, w, h) normalised, YOLO],
    meta=dict). Everything random comes from `rng`, so a seed fixes the frame."""
    look = look or Look()
    probs = MO.class_probs() if probs is None else probs
    w_px, h_px = look.size
    px = look.px_um
    f = max(1, math.ceil(px / look.fine_um - 1e-9))
    res = px / f
    W, H = w_px * px, h_px * px
    n = int(n_objects if n_objects is not None else rng.integers(look.n_objects[0], look.n_objects[1] + 1))
    objs = _place(rng, look, W, H, n, probs)
    specs = [sp for o in objs for sp in o["specs"]]
    mask = G.render_mask(specs, (W, H), res).astype(np.float32)
    ny, nx = mask.shape

    bg_name = None if look.bg_override else str(rng.choice(look.bg_names))
    bg = np.array(look.bg_override if look.bg_override else BACKGROUNDS[bg_name], float) + rng.normal(0, 0.02, 3)
    bg = bg.mean() + look.sat * (bg - bg.mean())
    yy = np.linspace(0, 1, ny, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, nx, dtype=np.float32)[None, :]
    lower = np.array(LOWER[bg_name] if bg_name else bg, float)
    lower = lower.mean() + look.sat * (lower - lower.mean())
    ang = math.radians(rng.uniform(-35, 35))                              # the skin shades from grey-green (nail side) to pink
    coord = (yy - 0.5) * math.cos(ang) + (xx - 0.5) * math.sin(ang)
    centre, width = rng.uniform(-0.1, 0.25), rng.uniform(0.08, 0.3)
    m = np.clip((coord - centre) / width * 0.5 + 0.5, 0, 1)
    m = (m * m * (3 - 2 * m))[..., None].astype(np.float32)
    base = bg[None, None, :] * (1 - m) + lower[None, None, :] * m
    base = base * (1.0 + _u(rng, look.bottom_shift) * yy)[..., None]
    ta = _u(rng, look.tint_amp)
    if ta > 0:                                                              # mild patchy colour cast on top
        ks = max(2, int(round(24.0 / res)))
        fld = np.stack([ndi.zoom(ndi.gaussian_filter(rng.normal(size=(ny // ks + 3, nx // ks + 3)), 6.0, mode="wrap"), ks, order=1)[:ny, :nx]
                        for _ in range(3)], -1)
        fld = (fld - fld.mean((0, 1))) / (fld.std((0, 1)) + 1e-9)
        base = base * (1.0 + ta * fld.astype(np.float32))
    contrast = _u(rng, look.contrast)
    ks = max(2, int(round(30.0 / res)))
    vis = ndi.zoom(ndi.gaussian_filter(rng.normal(size=(ny // ks + 3, nx // ks + 3)), 3.0, mode="wrap"), ks, order=1)[:ny, :nx]
    vis = np.clip(0.75 + 0.35 * (vis - vis.mean()) / (vis.std() + 1e-9), 0.25, 1.0)          # focus / depth: some loops fainter
    weight = (vis * (1.0 - 0.45 * np.clip((yy - 0.35) / 0.65, 0, 1))).astype(np.float32)       # fainter toward the proximal side
    ideal = base * (1.0 - (ABSORB * contrast)[None, None, :] * (mask * weight)[..., None])

    tex_amp = _u(rng, look.texture)
    if tex_amp > 0:
        kt = max(1, int(round(4.0 / res)))
        small = ndi.gaussian_filter(rng.normal(size=(ny // kt + 2, nx // kt + 2)), 25.0 / (res * kt), mode="wrap")
        tex = ndi.zoom(small, kt, order=1)[:ny, :nx]
        tex = ((tex - tex.mean()) / (tex.std() + 1e-9)).astype(np.float32)
        tint = 1.0 + tex_amp * tex[..., None] * np.array([1.0, 0.8, 0.6])[None, None, :]   # perfusion blotches: more in red
        ideal = ideal * tint
    st = _u(rng, look.striation)
    if st > 0:                                                   # streaks along the capillary direction (vertical)
        ks = max(1, int(round(2.0 / res)))
        sm = ndi.gaussian_filter(rng.normal(size=(ny // ks + 2, nx // ks + 2)), (30.0 / (res * ks), 1.5 / (res * ks)), mode="wrap")
        sm = ndi.zoom(sm, ks, order=1)[:ny, :nx]
        ideal = ideal * (1.0 + st * ((sm - sm.mean()) / (sm.std() + 1e-9)))[..., None].astype(np.float32)
    ill = _u(rng, look.illum)
    if ill > 0:
        a = rng.uniform(0, 2 * math.pi)
        X, Y = np.meshgrid(np.arange(nx, dtype=np.float32) - nx / 2, np.arange(ny, dtype=np.float32) - ny / 2)
        g = X * math.cos(a) + Y * math.sin(a)
        ideal = ideal * (1.0 + ill * g / (np.abs(g).max() * 2.0))[..., None]
        r2 = (X / (nx / 2)) ** 2 + (Y / (ny / 2)) ** 2
        ideal = ideal * (1.0 - 0.12 * rng.uniform(0, 1) * r2)[..., None]               # vignette

    sg = float(sigma_um) if sigma_um is not None else _lu(rng, look.sigma_um)
    ramp = _lu(rng, look.sigma_ramp)
    s_top, s_bot = sg * math.sqrt(ramp), sg / math.sqrt(ramp)
    blurred = []
    for s in (s_top, s_bot):
        blurred.append(np.stack([ndi.gaussian_filter(ideal[..., c], max(s, 0.05) / res, mode="nearest") for c in range(3)], -1))
    wy = yy[..., None]
    ideal = blurred[0] * (1 - wy) + blurred[1] * wy

    ideal = ideal[: (ny // f) * f, : (nx // f) * f]
    img = ideal.reshape(ny // f, f, nx // f, f, 3).mean(axis=(1, 3))
    hh, ww = img.shape[:2]

    # surface artifacts on the (oil) film: bright bubbles with a dark rim, dark specks, all slightly soft
    art = np.zeros_like(img)
    alpha = np.zeros(img.shape[:2], np.float32)
    for _ in range(int(rng.integers(look.bubbles[0], look.bubbles[1] + 1))):
        cy, cx = rng.uniform(0, hh), rng.uniform(0, ww)
        ry, rx = rng.uniform(2, 14), rng.uniform(2, 14)
        Y, X = np.ogrid[:hh, :ww]
        d = ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2
        core, ring = d <= 1.0, (d > 1.0) & (d <= 1.5)
        art[core] = 0.95
        alpha[core] = np.maximum(alpha[core], rng.uniform(0.6, 0.95))
        art[ring] = 0.25
        alpha[ring] = np.maximum(alpha[ring], 0.35)
    for _ in range(int(rng.integers(look.specks[0], look.specks[1] + 1))):
        cy, cx, r = int(rng.integers(0, hh)), int(rng.integers(0, ww)), rng.uniform(0.8, 2.2)
        Y, X = np.ogrid[:hh, :ww]
        m = (Y - cy) ** 2 + (X - cx) ** 2 <= r * r
        art[m] = 0.08
        alpha[m] = 0.9
    alpha = ndi.gaussian_filter(alpha, 0.8)
    img = img * (1 - alpha[..., None]) + art * alpha[..., None]
    for _ in range(int(rng.integers(look.glare[0], look.glare[1] + 1))):     # LED reflections: big, smooth, near white
        cy, cx = rng.uniform(-0.1, 1.1) * hh, rng.uniform(-0.1, 1.1) * ww
        ry, rx = rng.uniform(25, 70), rng.uniform(35, 120)
        th = rng.uniform(0, np.pi)
        Y, X = np.mgrid[:hh, :ww].astype(np.float32)
        u = (X - cx) * np.cos(th) + (Y - cy) * np.sin(th)
        v = -(X - cx) * np.sin(th) + (Y - cy) * np.cos(th)
        d = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        a = np.clip((1.06 - d) * 9.0, 0, 1)                       # flat, clipped core with a narrow soft rim
        halo = 0.18 * np.clip(1.6 - d, 0, 1)
        a = np.maximum(a, halo)
        img = img * (1 - a[..., None]) + np.array([1.08, 1.06, 0.98])[None, None, :] * a[..., None]

    photons = _lu(rng, look.photons)
    rn = _lu(rng, look.read_noise)
    img = rng.poisson(np.clip(img, 0, None) * photons) / photons + rng.normal(0, rn, img.shape)
    sh = _u(rng, look.sharpen)
    if sh > 0:
        img = img + sh * (img - ndi.gaussian_filter(img, (1.2, 1.2, 0)))
    u8 = np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)
    q = int(rng.integers(look.jpeg_q[0], look.jpeg_q[1] + 1))
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, "JPEG", quality=q)
    u8 = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

    boxes, meta_objs = [], []
    for o in objs:
        x0, y0, x1, y1 = MO.object_bbox(o["specs"])
        x0, x1 = max(0.0, x0 - 1.0), min(W, x1 + 1.0)
        yt, yb = max(0.0, H - y1 - 1.0), min(H, H - y0 + 1.0)       # to image coordinates (y down)
        if x1 - x0 < 4 or yb - yt < 8:
            continue
        cid = MO.CLASSES.index(o["cls"])
        boxes.append((cid, (x0 + x1) / 2 / W, (yt + yb) / 2 / H, (x1 - x0) / W, (yb - yt) / H))
        meta_objs.append(dict(cls=o["cls"], **o["truth"]))
    meta = dict(px_um=px, size=[w_px, h_px], sigma_um=sg, sigma_ramp=ramp, contrast=contrast, photons=photons,
                read_noise=rn, illum=ill, texture=tex_amp, bg=bg_name or "override", jpeg_q=q, sharpen=sh,
                sharpness=sharpness(u8), n_objects=len(boxes), density_per_mm=len(boxes) / (W / 1000.0),
                objects=meta_objs)
    return dict(img=u8, boxes=boxes, meta=meta)


def yolo_lines(boxes):
    return [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
