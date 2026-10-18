"""S0: real calibration -> results/scope_params.json.

    python run_s0.py --data data/s0 --out results/scope_params.json

Expected layout (see data/README.md):
    data/s0/slide/*.png        10 slide captures at the fixed finger-rest height, two sessions
    data/s0/flat/<level>/*.png blank card, >= 2 frames per exposure level, >= 3 levels, exposure locked
    data/s0/gradient/*.png     blank card at the working exposure (lighting gradient)
    data/s0/edge/*.png         optional: razor-edge profiles (black card edge) for sigma
    data/s0/contrast.csv       optional: columns condition,contrast  (loop contrast, oil / no_oil)
Missing parts leave the matching parameters flagged ASSUMED.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from capsim import calibration as C, intake
from capsim.scope import ScopeParams


def main(data, out, config):
    cfg = json.loads(Path(config).read_text()) if Path(config).exists() else {}
    data = Path(data)
    sc = ScopeParams()
    report = {}

    # 1. um per pixel
    if (data / "slide").exists():
        vals, profs = [], []
        for p in intake.list_images(data / "slide"):
            prof = C.tick_profile(intake.load_gray(p))
            profs.append(prof)
            try:
                v, se, n = C.umpx_from_ticks(prof, cfg.get("pitch_um", 10.0))
                vals.append(v)
            except ValueError as e:
                report.setdefault("slide_skipped", []).append(f"{p.name}: {e}")
        if vals:
            summ = C.umpx_summary(vals)
            report["um_per_px"] = summ
            sc.mark(um_per_px=summ["mean"])
            # 2. sigma from the tick (needs the real tick width)
            if cfg.get("tick_width_um"):
                sig = []
                for prof in profs:
                    try:
                        s, fit = C.sigma_from_ticks(prof, summ["mean"], cfg.get("pitch_um", 10.0), cfg["tick_width_um"])
                        sig.append(s)
                    except Exception:
                        pass
                if sig:
                    report["sigma_tick"] = dict(mean=float(np.mean(sig)), sd=float(np.std(sig, ddof=1)) if len(sig) > 1 else None, n=len(sig))
                    sc.mark(sigma_um=float(np.mean(sig)))

    if (data / "edge").exists() and "sigma_tick" not in report and sc.measured["um_per_px"]:
        sig = []
        for p in intake.list_images(data / "edge"):
            prof = C.tick_profile(intake.load_gray(p), angles=[0])
            sig.append(C.sigma_from_edge(prof, sc.um_per_px)[0])
        if sig:
            report["sigma_edge"] = dict(mean=float(np.mean(sig)), n=len(sig))
            sc.mark(sigma_um=float(np.mean(sig)))

    # 3. photon transfer
    if (data / "flat").exists():
        levels = {d.name: [intake.load_gray(p) for p in intake.list_images(d)] for d in sorted((data / "flat").iterdir()) if d.is_dir()}
        pt = C.photon_transfer(levels, bits=cfg.get("bits", 8), min_levels=cfg.get("min_levels", 3))
        report["photon_transfer"] = {k: v for k, v in pt.items() if k not in ("means", "vars")}
        if pt["linear"]:
            sc.mark(**C.to_forward(pt["g"], pt["read_noise_dn"], cfg.get("bits", 8)))
        else:
            report["photon_transfer"]["warning"] = ("not linear (r2 <= 0.98 or gain <= 0): camera likely applies gamma or "
                                                    "auto exposure. Photons and read noise left ASSUMED.")

    # 4. lighting gradient
    if (data / "gradient").exists():
        fl = [intake.load_gray(p) for p in intake.list_images(data / "gradient")]
        if fl:
            ig = C.illumination_gradient(np.mean(fl, axis=0), pixel_um=sc.um_per_px)
            report["illumination"] = ig
            sc.mark(illum_strength=ig["strength"], illum_angle_deg=ig["angle_deg"])

    # contrast (optional)
    cp = data / "contrast.csv"
    if cp.exists():
        df = intake.read_csv(cp, ["condition", "contrast"])
        for cond, col in (("no_oil", "contrast"), ("oil", "contrast_oil")):
            v = df[df.condition == cond].contrast
            if len(v):
                sc.mark(**{col: float(v.mean())})
        report["contrast"] = df.groupby("condition").contrast.agg(["mean", "std", "count"]).to_dict()

    sc.notes["s0_report"] = report
    sc.save(out)
    print(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}; still ASSUMED: {sc.assumed() or 'nothing'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/s0")
    ap.add_argument("--out", default="results/scope_params.json")
    ap.add_argument("--config", default="configs/s0_config.json")
    a = ap.parse_args()
    main(a.data, a.out, a.config)
