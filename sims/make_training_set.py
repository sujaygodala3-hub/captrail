"""Generate a labelled synthetic nailfold training set in YOLO format (class order = the Capillary Dataset's).

  python make_training_set.py --n 2000 --out data/synth --look dataset
  python make_training_set.py --n 500 --look celestron --out data/synth_celestron
  python make_training_set.py --n 2000 --look configs/look_fitted.json --profile results/dataset_profile.json --balance 0.4

Writes images/{train,val,test}/*.jpg, labels/{split}/*.txt, data.yaml (ready for `yolo detect train data=...`),
meta.jsonl (per-image truth: PSF sigma, contrast, noise, every object's class and true widths) and manifest.json.
Splits are by image index with independent seeds, so train and test never share a frame. Synthetic data
should pre-train or augment, not replace, real images: always validate on REAL images.
"""
import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from capsim import morphology as MO, render_rgb as R

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--out", default="data/synth")
ap.add_argument("--look", default="dataset", help="dataset | celestron | path to a Look json (from fit_look.py)")
ap.add_argument("--profile", default=None, help="dataset_profile.json: use its class counts and box statistics")
ap.add_argument("--balance", type=float, default=0.0, help="0 = dataset class frequency, 1 = uniform")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1))
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--px-um", type=float, default=None, help="override um per pixel (use your S0 value)")
ap.add_argument("--scope", default=None, help="results/scope_params.json: use its measured um/px and sigma")
a = ap.parse_args()

if a.look in ("dataset", "celestron"):
    look = getattr(R.Look, a.look)()
else:
    look = R.Look.load(a.look)
if a.scope:
    from capsim.scope import ScopeParams
    sc = ScopeParams.load(a.scope)
    from dataclasses import replace
    look = replace(look, px_um=sc.um_per_px, sigma_um=(max(0.5, sc.sigma_um * 0.7), sc.sigma_um * 1.8))
if a.px_um:
    from dataclasses import replace
    look = replace(look, px_um=a.px_um)
counts = None
if a.profile:
    counts = json.loads(Path(a.profile).read_text()).get("class_counts")
probs = MO.class_probs(counts, a.balance)

out = Path(a.out)
cut = np.cumsum(a.split) / sum(a.split)
splits = ["train", "val", "test"]
for s in splits:
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)


def one(i):
    sp = splits[int(np.searchsorted(cut, (i + 0.5) / a.n))]
    o = R.render_scene(np.random.default_rng([a.seed, i]), look, probs)
    name = f"syn_{i:06d}"
    Image.fromarray(o["img"]).save(out / "images" / sp / f"{name}.jpg", quality=97)
    (out / "labels" / sp / f"{name}.txt").write_text("\n".join(R.yolo_lines(o["boxes"])) + "\n")
    return dict(name=name, split=sp, **o["meta"])


if __name__ == "__main__":
    with Pool(a.workers) as pool:
        metas = pool.map(one, range(a.n), chunksize=8)
    with open(out / "meta.jsonl", "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                   f"nc: {len(MO.CLASSES)}\nnames: {MO.CLASSES}\n")
    cc = {k: 0 for k in MO.CLASSES}
    for m in metas:
        for o in m["objects"]:
            cc[o["cls"]] += 1
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git = ""
    (out / "manifest.json").write_text(json.dumps(dict(n=a.n, seed=a.seed, look=a.look, px_um=look.px_um, class_counts=cc,
                                                       balance=a.balance, profile=a.profile, git=git, argv=sys.argv), indent=2))
    print(f"{a.n} images -> {out}  classes {cc}")
