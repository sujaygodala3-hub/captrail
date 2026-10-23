"""Check the real-frame pipeline (analyze_session.analyse_frame) on simulated frames built to MATCH the
measured session statistics (blur, lumen width, contrast, lighting, noise), where the truth is known.

  python validate_session_pipeline.py --sigma 1.9 --lumen 6 --total 22 --contrast 0.095 --illum 0.16 --out results/validation.json
"""
import argparse
import json

import numpy as np

from capsim import geometry as G, optics as O
from analyze_session import analyse_frame


def frame(rng, sigma, lumen, total, contrast, illum, n=8, apical_extra=2.0):
    W, H = 640.0, 480.0
    spacing = W / n
    specs = []
    for i in range(n):
        sp = G.make_loop(lumen, lumen + apical_extra, total, limb_len=rng.uniform(90, 140))
        sp.cx = (i + 0.5) * spacing + rng.uniform(-0.15, 0.15) * spacing
        sp.y_base = H - rng.uniform(140, 260) - sp.limb_len
        specs.append(sp)
    mask = G.render_mask(specs, (W, H), 0.5)
    g = O.forward(mask, 0.5, sigma_um=sigma, contrast=contrast, illum_strength=illum, illum_angle_deg=rng.uniform(0, 360),
                  pixel_um=1.0, photons=50000, read_noise=0.002, exposure=0.6, rng=rng, texture_amp=0.01)
    rgb = np.stack([np.clip(g * 1.45, 0, 1), g, np.clip(g * 0.62, 0, 1)], -1)
    truth = dict(n=n, apical=float(np.mean([G.true_apical_width(s) for s in specs])))
    return (rgb * 255).round().astype(np.uint8), truth


def main(a):
    rng = np.random.default_rng(a.seed)
    rows = []
    for _ in range(a.frames):
        rgb, tr = frame(rng, a.sigma, a.lumen, a.total, a.contrast, a.illum)
        q, loops, sig, gaps, ap = analyse_frame(rgb)
        lw = [v for l in loops for v in (l["limb_left"], l["limb_right"]) if np.isfinite(v)]
        rows.append(dict(sigma_fit=sig.get("sigma_px", np.nan), lumen_fit=float(np.median(sig["widths_px"])) if sig.get("widths_px") else np.nan,
                         contrast_fit=sig.get("contrast", np.nan), limb_fwhm=float(np.median(lw)) if lw else np.nan,
                         apical=float(np.nanmedian([l["apical"] for l in loops])) if loops else np.nan,
                         total=float(np.nanmedian([l["total"] for l in loops])) if loops else np.nan,
                         n_apices=len(ap), n_true=tr["n"], apical_true=tr["apical"]))
    import pandas as pd
    D = pd.DataFrame(rows)
    res = dict(truth=dict(sigma=a.sigma, lumen=a.lumen, total=a.total, contrast=a.contrast, apical=float(D.apical_true.mean())),
               recovered={c: dict(mean=float(D[c].mean()), sd=float(D[c].std(ddof=1))) for c in ("sigma_fit", "lumen_fit", "contrast_fit", "limb_fwhm", "apical", "total", "n_apices")},
               frames=a.frames)
    res["bias"] = dict(sigma_pct=100 * (D.sigma_fit.mean() / a.sigma - 1), lumen_pct=100 * (D.lumen_fit.mean() / a.lumen - 1),
                       limb_fwhm_vs_lumen_pct=100 * (D.limb_fwhm.mean() / a.lumen - 1), total_pct=100 * (D.total.mean() / a.total - 1),
                       apical_pct=100 * (D.apical.mean() / D.apical_true.mean() - 1), apex_recall=float((D.n_apices / D.n_true).mean()))
    print(json.dumps(res, indent=2))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.9)
    ap.add_argument("--lumen", type=float, default=6.0)
    ap.add_argument("--total", type=float, default=22.0)
    ap.add_argument("--contrast", type=float, default=0.095)
    ap.add_argument("--illum", type=float, default=0.16)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
