"""Analyse one real capture session (a folder of scan-page PNGs named <who>-<finger>-<NN><n|e|g>.png).

  python analyze_session.py path/to/session --out results/session_name [--um-per-px 1.6]

Sorts raw from processed frames by their pixel statistics (file suffixes can be wrong), masks glare,
finds the loop apexes in the distal row, measures limb, apical and total width per loop, fits the blur
(PSF sigma) per frame with a shared-sigma model, and reports repeatability across repeat captures.
Everything is in PIXELS unless --um-per-px is given (and that must come from the ruler, S0).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from capsim import analyze as A, measure as M, realdata as RD, stats as S

NAME = re.compile(r"(?P<who>[a-z0-9]+)-(?P<hand>[LR])?(?P<finger>[a-z]+)-(?P<cap>\d+)-?(?P<suffix>[neg])$")   # sujay-ring-05e, person1-Lindex-1-n
FINGER_ALIAS = {"pink": "pinky", "index": "pointer"}


def analyse_frame(rgb):
    mask = RD.glare_mask(rgb)
    g = RD.clean_green(rgb, mask)
    q = RD.quality(rgb, mask)
    flat = M.flatten(g, 1.0, fast=True)
    apices, _ = A.find_apices(g, 1.0, min_prominence_um=8.0, min_sep_um=14.0, min_area_um2=120.0, min_height_um=25.0)
    H, W = g.shape
    near_glare = __import__("scipy").ndimage.binary_dilation(mask, iterations=12)
    loops, profs = [], []
    for ax, ay in apices:
        if ax < 12 or ax > W - 12 or near_glare[min(H - 1, int(ay)), min(W - 1, int(ax))]:
            continue
        m, pp = RD.measure_loop(flat, ax, ay)
        loops.append(m)
        profs += pp
    sig = RD.fit_limbs_shared_sigma([p for p, _ in profs], [c for _, c in profs]) if len(profs) >= 4 else dict(sigma_px=np.nan, n=len(profs))
    xs = np.sort([l["apex_x"] for l in loops])
    gaps = np.diff(xs) if len(xs) > 1 else np.array([])
    return q, loops, sig, gaps, apices


def main(folder, out, um_per_px=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(folder).glob("*.png"))
    frames, loops_all = [], []
    vis = {}
    named = [p for p in files if NAME.search(p.stem)]
    whos = {NAME.search(p.stem)["who"] for p in named}
    hands = any(NAME.search(p.stem)["hand"] for p in named)
    multi = len(whos) > 1 or hands          # several people or hands: a 'finger' is person-hand-digit
    if not named:                      # any naming: each file is its own capture of an 'unknown' finger
        print(f"note: no file matches <who>-<finger>-<NN><n|e|g>; treating {len(files)} files as raw captures of one finger")
    for i, p in enumerate(files):
        mt = NAME.search(p.stem)
        if not mt and named:
            continue
        mt = mt or dict(who="x", hand=None, finger="unknown", cap=str(i + 1), suffix="n")
        rgb = np.asarray(Image.open(p).convert("RGB"))
        kind = RD.classify(rgb)
        q, loops, sig, gaps, apices = analyse_frame(rgb)
        digit = FINGER_ALIAS.get(mt["finger"], mt["finger"])
        finger = f"{mt['who']}-{mt['hand'] or ''}{digit}" if multi else digit
        row = dict(file=p.name, who=mt["who"], hand=mt["hand"] or "", digit=digit, finger=finger, capture=int(mt["cap"]), suffix=mt["suffix"], kind=kind,
                   suffix_matches=(mt["suffix"] == "n" and kind == "raw") or (mt["suffix"] == "e" and kind == "enhanced") or (mt["suffix"] == "g" and kind == "grey"),
                   n_apices=len(apices), n_loops=len(loops), n_resolved=int(sum(np.isfinite(l["limb_left"]) for l in loops)),
                   median_gap_px=float(np.median(gaps)) if len(gaps) else np.nan,
                   min_gap_px=float(np.min(gaps)) if len(gaps) else np.nan,
                   sigma_px=sig.get("sigma_px", np.nan), sigma_n_limbs=sig.get("n", 0), sigma_at_bound=sig.get("at_bound", None),
                   lumen_width_px_fit=float(np.median(sig["widths_px"])) if sig.get("widths_px") else np.nan,
                   lumen_contrast_fit=sig.get("contrast", np.nan), illum_strength=RD.illumination_strength(rgb[..., 1].astype(float), RD.glare_mask(rgb))[0], **q)
        for k in ("limb_left", "limb_right", "apical", "total", "depth_frac"):
            v = [l[k] for l in loops if np.isfinite(l[k])]
            row[f"median_{k}_px"] = float(np.median(v)) if v else np.nan
        frames.append(row)
        for l in loops:
            loops_all.append(dict(file=p.name, who=mt["who"], hand=mt["hand"] or "", digit=digit, finger=finger, capture=int(mt["cap"]), kind=kind, **l))
        vis[p.name] = (rgb, apices, loops)
    F = pd.DataFrame(frames)
    # one PRIMARY (unprocessed) frame per finger and capture: the 'n' file unless its pixels say it is the
    # processed view, then the other one. The other file of the pair is the SECONDARY (processed) copy.
    F["primary"] = False
    for _, g in F.groupby(["finger", "capture"]):
        cand = g[(g.suffix == "n") & (g.kind == "raw")]
        if cand.empty:
            cand = g[g.kind == "raw"]
        if cand.empty:          # no unprocessed frame for this capture (e.g. an enhanced-only zip): measure what exists
            cand = g[g.suffix == "n"] if (g.suffix == "n").any() else g
        if not cand.empty:
            F.loc[cand.index[0], "primary"] = True
    L = pd.DataFrame(loops_all)
    L["primary"] = L.file.isin(F.loc[F.primary, "file"])
    F.to_csv(out / "frames.csv", index=False)
    L.to_csv(out / "loops.csv", index=False)
    summary = summarise(F, L, um_per_px)
    (out / "session_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    figures(F, L, vis, out)
    print(json.dumps(summary, indent=2, default=float)[:4000])
    return F, L, summary


def usable(F):
    return F[F.primary & (F.n_resolved >= 3) & (F.glare_frac < 0.30)]


def summarise(F, L, um_per_px):
    raw = F[F.primary]
    U = usable(F)
    LU = L[L.file.isin(U.file)]
    s = dict(n_files=int(len(F)), kinds=F.kind.value_counts().to_dict(),
             suffix_mismatches=F.loc[~F.suffix_matches, ["file", "kind"]].to_dict("records"),
             n_primary=int(len(raw)), n_usable=int(len(U)), usable_files=U.file.tolist(),
             unusable_primary=F.loc[F.primary & ~F.file.isin(U.file), ["file", "n_resolved", "glare_frac", "sharpness"]].to_dict("records"),
             loops_detected_primary=int(L[L.primary].shape[0]), loops_resolved_usable=int(np.isfinite(LU.limb_left).sum()))
    def q(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if not len(x):
            return {}
        m, lo, hi, n = S.mean_ci(x)
        return dict(mean=m, ci95=[lo, hi], median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)), sd=float(x.std(ddof=1)) if n > 1 else np.nan, n=n)
    limbs = np.r_[LU.limb_left.values, LU.limb_right.values]
    s["px"] = dict(limb_fwhm=q(limbs), apical_fwhm=q(LU.apical), total=q(LU.total), sigma_px_per_frame=q(U.sigma_px),
                   lumen_width_fit=q(U.lumen_width_px_fit), lumen_contrast_fit=q(U.lumen_contrast_fit), illum_strength=q(U.illum_strength), gap_between_apices=q(U.median_gap_px), loops_per_frame=q(U.n_apices),
                   depth_frac=q(LU.depth_frac))
    s["quality_raw"] = dict(glare_frac=q(raw.glare_frac), sharpness=q(raw.sharpness), noise_dn=q(raw.noise_dn),
                            clipped_frac=q(raw.clipped_frac), early_captures_usable=int(U[U.capture <= 3].shape[0]),
                            late_captures_usable=int(U[U.capture >= 4].shape[0]),
                            early_raw=int(raw[raw.capture <= 3].shape[0]), late_raw=int(raw[raw.capture >= 4].shape[0]))
    rep = {}
    for metric in ("median_apical_px", "median_limb_left_px", "median_total_px", "n_apices", "median_gap_px", "sigma_px"):
        per = {f: g[metric].dropna().values for f, g in U.groupby("finger")}
        cvs = [100 * v.std(ddof=1) / v.mean() for v in per.values() if len(v) >= 2 and v.mean() > 0]
        k = min((len(v) for v in per.values() if len(v) >= 2), default=0)
        mat = np.array([v[:k] for v in per.values() if len(v) >= k]) if k >= 2 else None
        icc_all = RD.icc_oneway_unbalanced(list(per.values()))
        sd_w = np.sqrt(np.mean([v.var(ddof=1) for v in per.values() if len(v) >= 2])) if cvs else np.nan
        rep[metric] = dict(within_finger_cv_pct=float(np.mean(cvs)) if cvs else np.nan, fingers=len(cvs),
                           frames_used=int(sum(len(v) for v in per.values())), within_finger_sd=float(sd_w),
                           repeatability_coefficient=float(1.96 * np.sqrt(2) * sd_w) if np.isfinite(sd_w) else None,
                           icc_1_unbalanced=icc_all if np.isfinite(icc_all) else None,
                           per_finger_mean={f: float(np.mean(v)) for f, v in per.items() if len(v)})
    s["repeatability"] = rep
    pairs = []
    for (f, c), g in F.groupby(["finger", "capture"]):
        r, e = g[g.primary], g[~g.primary & (g.kind != "grey")]
        if len(r) == 1 and len(e) == 1:
            pairs.append(dict(finger=f, capture=c, **{f"raw_{k}": float(r[k].iloc[0]) for k in ("median_apical_px", "median_limb_left_px", "n_apices", "sigma_px", "sharpness", "noise_dn")},
                              **{f"enh_{k}": float(e[k].iloc[0]) for k in ("median_apical_px", "median_limb_left_px", "n_apices", "sigma_px", "sharpness", "noise_dn")}))
    P = pd.DataFrame(pairs)
    if len(P):
        s["raw_vs_enhanced"] = {k: dict(n=int(P[[f"raw_{k}", f"enh_{k}"]].dropna().shape[0]),
                                        mean_raw=float(P[f"raw_{k}"].mean()), mean_enh=float(P[f"enh_{k}"].mean()),
                                        mean_diff_enh_minus_raw=float((P[f"enh_{k}"] - P[f"raw_{k}"]).mean()))
                                for k in ("median_apical_px", "median_limb_left_px", "n_apices", "sigma_px", "sharpness", "noise_dn")}
    if um_per_px:
        s["um"] = {k: {kk: (vv * um_per_px if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("mean", "median", "p10", "p90", "sd")}
                   for k, v in s["px"].items() if k not in ("loops_per_frame", "depth_frac")}
        s["um_per_px_used"] = um_per_px
    return s


def figures(F, L, vis, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    U = usable(F)
    names = U.sort_values(["finger", "capture"]).file.tolist()[:12]
    if names:
        cols = 4
        rows = int(np.ceil(len(names) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.1))
        for ax in np.ravel(axs):
            ax.axis("off")
        for ax, n in zip(np.ravel(axs), names):
            rgb, apices, loops = vis[n]
            ax.imshow(rgb)
            for l in loops:
                c = "lime" if np.isfinite(l["limb_left"]) else "yellow"
                ax.plot(l["apex_x"], l["apex_y"], "o", ms=7, mfc="none", mec=c, mew=1.5)
            ax.set_title(n, fontsize=8)
        fig.suptitle("Detected loop apexes on usable raw frames (green = both limbs resolved, yellow = apex only)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig_detections.png", dpi=110)
        plt.close(fig)
    LU = L[L.file.isin(U.file)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (k, lab) in zip(axs, (("limb", "limb FWHM (px)"), ("apical", "apical FWHM (px)"), ("total", "total loop width (px)"))):
        v = np.r_[LU.limb_left, LU.limb_right] if k == "limb" else LU[k].values
        v = v[np.isfinite(v)]
        ax.hist(v, bins=25, color="#4a7ab5", edgecolor="white")
        ax.axvline(np.median(v), color="k", ls="--", lw=1)
        ax.set_xlabel(lab)
        ax.set_ylabel("loops")
        ax.set_title(f"median {np.median(v):.1f} px, n={len(v)}")
    fig.tight_layout()
    fig.savefig(out / "fig_widths_px.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    order = ["thumb", "pointer", "middle", "ring", "pinky"]
    for i, f in enumerate(order):
        g = U[U.finger == f]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.15, 0.15, max(len(g), 1))[:len(g)], g.median_apical_px, color="#4a7ab5")
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("median apical FWHM per capture (px)")
    ax.set_title("Repeat captures per finger (raw, usable frames)")
    fig.tight_layout()
    fig.savefig(out / "fig_repeatability.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/session")
    ap.add_argument("--um-per-px", type=float, default=None)
    a = ap.parse_args()
    main(a.folder, a.out, a.um_per_px)
