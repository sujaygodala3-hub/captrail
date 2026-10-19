"""Measure real captures with the same estimators the simulations validate.

    python analyze_captures.py data/captures --apex up --scope results/scope_params.json --out results/analysis

Writes results/analysis/images.csv (one row per image: quality gate, loops per mm) and loops.csv
(one row per loop: apical, limb and total width in um). Use the apical widths as `tool_um` for S5 and
the repeated-capture widths for S7. um per pixel comes from the scope file and must be MEASURED (S0).
"""
import argparse
from pathlib import Path

import pandas as pd

from capsim import intake
from capsim.analyze import analyze_image
from capsim.scope import ScopeParams


def main(folder, apex, scope_path, out, recursive):
    sc = ScopeParams.load(scope_path)
    if not sc.measured.get("um_per_px"):
        print("WARNING: um_per_px is ASSUMED, not measured. Widths below are NOT in real micrometers. Run run_s0.py first.")
    files = (sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in intake.IMG_EXT and (p.stem.endswith("_raw") or not any(q.stem.endswith("_raw") for q in p.parent.iterdir())))
             if recursive else intake.list_images(folder))
    imgs, loops = [], []
    for p in files:
        im = intake.load_gray(p)
        info, lp = analyze_image(im, sc.um_per_px, apex)
        info["file"] = str(p.relative_to(folder)) if recursive else p.name
        info["um_per_px_measured"] = bool(sc.measured.get("um_per_px"))
        imgs.append(info)
        if len(lp):
            lp.insert(0, "file", info["file"])
            loops.append(lp)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(imgs).to_csv(out / "images.csv", index=False)
    (pd.concat(loops) if loops else pd.DataFrame()).to_csv(out / "loops.csv", index=False)
    print(f"{len(files)} images, {sum(len(l) for l in loops)} loops -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--apex", default="up", choices=["up", "down", "left", "right", "auto"],
                    help="direction the loop apexes point in the saved image (cuticle side)")
    ap.add_argument("--scope", default="results/scope_params.json")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args()
    main(a.folder, a.apex, a.scope, a.out, a.recursive)
