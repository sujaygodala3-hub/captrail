"""Profile a local copy of the Capillary Dataset (or any YOLO-format nailfold set) so the simulator can be
matched to it: class counts, boxes per image, box size and position, image size, colour and sharpness.

Download first (about 205 MB):  huggingface-cli download HanaNguyen/Capillary-Dataset --repo-type dataset \
    --include "Morphology_detection/*" --local-dir data/capillary_dataset
Then:  python profile_dataset.py data/capillary_dataset/Morphology_detection/images data/capillary_dataset/Morphology_detection/labels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import domain_gap as DG, morphology as MO
from .intake import IMG_EXT

Q = (5, 25, 50, 75, 95)


def _q(x):
    x = np.asarray(x, float)
    return {f"p{q}": float(np.percentile(x, q)) for q in Q} if len(x) else {}


def read_yolo(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if len(p) >= 5:
            rows.append((int(float(p[0])), *map(float, p[1:5])))
    return rows


def profile_dataset(img_dir, label_dir=None, max_images=None, seed=0):
    from PIL import Image
    fs = sorted(p for p in Path(img_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not fs:
        raise FileNotFoundError(f"no images under {img_dir}")
    rng = np.random.default_rng(seed)
    if max_images and len(fs) > max_images:
        fs = [fs[i] for i in sorted(rng.permutation(len(fs))[:max_images])]
    sizes, feats, per_img, boxes = {}, [], [], []
    lab_index = {q.stem: q for q in Path(label_dir).rglob("*.txt")} if label_dir else {}      # works for flat or train/val/test layouts
    for p in fs:
        im = Image.open(p).convert("RGB")
        sizes[f"{im.size[0]}x{im.size[1]}"] = sizes.get(f"{im.size[0]}x{im.size[1]}", 0) + 1
        feats.append(DG.features(np.asarray(im)))
        if label_dir:
            rows = read_yolo(lab_index[p.stem]) if p.stem in lab_index else []
            per_img.append(len(rows))
            boxes += rows
    F = np.stack(feats)
    out = dict(n_images=len(fs), image_sizes=sizes, features={n: _q(F[:, j]) for j, n in enumerate(DG.FEATURES)})
    if boxes:
        B = np.array(boxes, float)
        counts = {MO.CLASSES[k] if k < len(MO.CLASSES) else str(k): int((B[:, 0] == k).sum()) for k in np.unique(B[:, 0]).astype(int)}
        out.update(class_counts=counts, boxes_per_image=dict(mean=float(np.mean(per_img)), **_q(per_img)),
                   box_w=_q(B[:, 3]), box_h=_q(B[:, 4]), box_cy=_q(B[:, 2]), box_top=_q(B[:, 2] - B[:, 4] / 2),
                   per_class={MO.CLASSES[int(k)]: dict(w=_q(B[B[:, 0] == k, 3]), h=_q(B[B[:, 0] == k, 4]))
                              for k in np.unique(B[:, 0]) if int(k) < len(MO.CLASSES)})
    return out


def infer_px_um(profile, loop_total_um=40.0, size_w=640):
    """um per pixel implied by the median hairpin box width, ASSUMING a hairpin loop is about `loop_total_um`
    wide. A guess: confirm with the S0 ruler on the real scope."""
    w = profile.get("per_class", {}).get("hairpin", {}).get("w", {}).get("p50") or profile.get("box_w", {}).get("p50")
    return None if not w else float(loop_total_um / (w * size_w))
