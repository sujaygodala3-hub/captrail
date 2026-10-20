"""Fit the colour simulator to real frames: python fit_look.py REAL_IMAGES_DIR [--base dataset|celestron] [--out configs/look_fitted.json]
Reports the two-sample AUC on a held-out half before and after (0.5 = indistinguishable)."""
import argparse
from pathlib import Path

import numpy as np

from capsim import adapt, domain_gap as DG, render_rgb as R
from capsim.intake import IMG_EXT

ap = argparse.ArgumentParser()
ap.add_argument("real_dir")
ap.add_argument("--base", default="dataset", choices=["dataset", "celestron"])
ap.add_argument("--out", default="configs/look_fitted.json")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--n-synth", type=int, default=30)
ap.add_argument("--max-real", type=int, default=400)
ap.add_argument("--size", type=int, nargs=2, default=(224, 168))
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
fs = sorted(p for p in Path(a.real_dir).rglob("*") if p.suffix.lower() in IMG_EXT)[: a.max_real * 4]
rng = np.random.default_rng(0)
fs = [fs[i] for i in rng.permutation(len(fs))[: a.max_real]]
half = len(fs) // 2
fit_imgs, held = [DG.load_rgb(p) for p in fs[:half]], [DG.load_rgb(p) for p in fs[half:]]
base = getattr(R.Look, a.base)()
size = tuple(a.size)
look, info = adapt.fit_look(fit_imgs, base, rounds=a.rounds, n_synth=a.n_synth, size=size, workers=a.workers)
Fh = DG.feature_matrix(held, size)
for name, lk in (("before", base), ("after", look)):
    sy = adapt.synth_feats(lk, 100, 5000, size)
    print(name, "held-out AUC", round(DG.two_sample(Fh, sy)["auc"], 3))
look.save(a.out)
print("objective", round(info["objective"], 3), "->", a.out)
