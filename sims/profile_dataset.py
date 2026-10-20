"""Profile a YOLO-format nailfold dataset: python profile_dataset.py IMAGES [LABELS] [--out results/dataset_profile.json]"""
import argparse
import json
from pathlib import Path

from capsim import dataset_profile as DP

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("labels", nargs="?")
ap.add_argument("--max-images", type=int, default=None)
ap.add_argument("--out", default="results/dataset_profile.json")
a = ap.parse_args()
prof = DP.profile_dataset(a.images, a.labels, a.max_images)
prof["px_um_guess_if_loop_40um"] = DP.infer_px_um(prof)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(prof, indent=2))
print(json.dumps({k: prof[k] for k in prof if k not in ("features", "per_class")}, indent=2))
print("->", a.out)
