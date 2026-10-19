"""Load real captures from the layout in data/README.md and compute QC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_gray(path, bits=None):
    """Read an image as a float 2D array in 8-bit digital numbers (0 to 255).

    Colour frames use the GREEN channel only (a luminance mix of independent channels would change the
    noise variance but not the mean and bias the photon-transfer gain). Scaling depends on the file's
    dtype, never on the frame's own maximum: uint16 is divided by 257, or by (2**bits - 1)/255 when the
    sensor is `bits`-bit stored in 16 bits (for example bits=12). Floats are assumed already in 0 to 255."""
    from skimage import io
    raw = io.imread(str(path))
    a = raw.astype(float)
    if a.ndim == 3:
        a = a[..., 1] if a.shape[2] >= 3 else a[..., 0]
    if raw.dtype.kind in "ui" and raw.dtype.itemsize == 2:
        a = a * 255.0 / ((2 ** bits - 1) if bits else 65535.0)
    return a


def list_images(folder, raw_only=True):
    """Image files in a folder. The scan page saves '<name>_raw.png' plus a processed view;
    for measurement use the raw one."""
    fs = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)
    if raw_only and any(p.stem.endswith("_raw") for p in fs):
        fs = [p for p in fs if p.stem.endswith("_raw")]
    return fs


def qc_table(folder, **gate_kw):
    from .measure import quality_gate
    rows = []
    for p in list_images(folder):
        img = load_gray(p)
        rows.append(dict(file=p.name, h=img.shape[0], w=img.shape[1], **quality_gate(img, **gate_kw)))
    return pd.DataFrame(rows)


def read_csv(path, required):
    df = pd.read_csv(path)
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{path} is missing columns {miss}")
    return df
