"""One replicable pipeline for every capture zip in a folder.

    python batch_run.py ~/Desktop/Coding/Caprprp --out results/batch            # quick sims (default)
    python batch_run.py ~/Desktop/Coding/Caprprp --out results/batch --full     # full-length sims
    python batch_run.py ~/Desktop/Coding/Caprprp --out results/batch --sims none

What it does, identically, for every zip (or folder of images) it finds:
  1. hashes the input (sha256) and extracts it (macOS __MACOSX/ and dot files dropped)
  2. sorts every image by its PIXELS into capillary / ruler / blank-card frames
  3. ruler zips  -> um/px (S0) and the blur at the ruler's zoom
     blank zips  -> camera noise spectrum: per-pixel noise and the blotchy (correlated) part
  4. capillary zips -> analyze_session (glare mask, loops, widths, shared-sigma blur) and then
     the same summaries per FINGER, per CONDITION (raw / enhanced / grey) and per CAPTURE PHASE
  5. builds scope files from what was measured (blur, contrast, lighting, noise spectrum, skin texture)
     for each scale scenario, each finger, and a white-noise-only ablation, and runs the same sims on each
  6. writes a manifest (input hash, code hash, library versions, output hashes) so a rerun or an
     identical input can be checked byte for byte; zips with the same hash are compared automatically

Everything is seeded; nothing depends on wall-clock time or file order. No images are copied into results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from capsim import calibration as C, geometry as G, optics as O, realdata as RD   # noqa: E402
from capsim.scope import ScopeParams   # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SKIP = re.compile(r"^(captrail_results|_|\.)")
SCALES = {"anat_central": 1.6, "anat_low": 1.3, "anat_high": 2.0}   # um/px, anatomical (Piotto 2016) until the ruler is confirmed
QUICK_STEPS = ["s1", "s2", "s4", "s6", "s7"]
FULL_STEPS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
HASHED = {".csv", ".json"}
NOT_HASHED = {"manifest.json", "timing.json", "input_state.json", "session_state.json"}


# ---------------------------------------------------------------- io helpers
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


ANALYSIS_VERSION = "1"   # bump when batch_run's analysis functions change (group tables, texture, scopes, ruler, blank)
STEP_COST = {"s1": 50, "s2": 95, "s3": 60, "s4": 50, "s5": 25, "s6": 18, "s7": 12, "s8": 150}   # s, one core busy among 4


def files_hash(names):
    h = hashlib.sha256()
    for p in names:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def analysis_hash():
    """What the cached analysis units depend on: capsim, the session analyzer and ANALYSIS_VERSION."""
    return _key(files_hash(sorted((ROOT / "capsim").glob("*.py")) + [ROOT / "analyze_session.py"]), ANALYSIS_VERSION)


def sims_hash():
    """What a cached sim step depends on: capsim, run_sims and its plots."""
    return files_hash(sorted((ROOT / "capsim").glob("*.py")) + [ROOT / "run_sims.py", ROOT / "plots_s.py"])


def code_hash():
    h = hashlib.sha256()
    for p in sorted(list((ROOT / "capsim").glob("*.py")) + [ROOT / n for n in ("batch_run.py", "analyze_session.py", "run_sims.py", "plots_s.py")]):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def versions():
    import scipy
    import skimage
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                skimage=skimage.__version__, platform=platform.platform())


def discover(folder):
    folder = Path(folder).expanduser()
    out = []
    for p in sorted(folder.iterdir(), key=lambda q: q.name):
        if SKIP.match(p.name):
            continue
        if p.is_file() and p.suffix.lower() == ".zip":
            out.append(p)
        elif p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir() if q.is_file()):
            out.append(p)
    return out


def extract(src, dest):
    """Unzip (or copy) images only, flattened, sorted. Returns the folder."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if Path(src).is_dir():
        for q in sorted(Path(src).iterdir()):
            if q.is_file() and q.suffix.lower() in IMG_EXT and not q.name.startswith("."):
                shutil.copyfile(q, dest / q.name)
        return dest
    with zipfile.ZipFile(src) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            name = info.filename
            base = Path(name).name
            if info.is_dir() or "__MACOSX" in name or base.startswith(".") or Path(base).suffix.lower() not in IMG_EXT:
                continue
            with z.open(info) as f, open(dest / base, "wb") as g:
                shutil.copyfileobj(f, g)
    return dest


def input_hash(src):
    if Path(src).is_file():
        return sha256(src)
    h = hashlib.sha256()
    for q in sorted(Path(src).iterdir()):
        if q.is_file() and q.suffix.lower() in IMG_EXT:
            h.update(q.name.encode())
            h.update(sha256(q).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- image type
def classify_image(rgb):
    """'ruler' | 'blank' | 'capillary', from pixels only.

    ruler: the most periodic band of rows repeats strongly at one pitch (mean autocorrelation over the first
    five periods >= 0.6 with >= 10 periods in view), as a stage micrometer does and a row of capillary loops
    does not; blank: almost no structure; otherwise capillary."""
    from scipy import ndimage as ndi
    g = np.asarray(rgb, float)[..., 1]
    try:
        r = C.slide_pitch_px(g, angles=[0.0])
        ac, nper = r["autocorr"], r["n_periods"]
    except Exception:
        ac, nper = 0.0, 0
    rel = g / np.maximum(ndi.gaussian_filter(g, 30), 1e-6) - 1
    dark = float((ndi.gaussian_filter(rel, 1.5) < -0.06).mean())
    m = dict(tick_autocorr=float(ac), periods=int(nper), std=float(g.std()), dark_frac=dark)
    if float(g.std()) < 15 and dark < 0.002:
        return "blank", m
    if ac >= 0.6 and nper >= 10:
        return "ruler", m
    return "capillary", m


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


# ---------------------------------------------------------------- calibration zips
def run_ruler(files, out, pitch_um):
    rows = []
    for p in files:
        g = load_rgb(p)[..., 1].astype(float)
        r = dict(file=p.name)
        try:
            v, se, info = C.umpx_from_slide(g, pitch_um)
            r.update(um_per_px=v, se=se, pitch_px=info["pitch_px"], tilt_deg=info["angle_deg"], periods=info["n_periods"],
                     long_tick_px=C.long_tick_spacing_px(g))
            r["long_over_10_fine"] = r["long_tick_px"] / (10 * info["pitch_px"])
        except Exception as e:
            r["error"] = str(e)
        try:
            v_old = C.umpx_from_ticks(C.tick_profile(g), pitch_um)[0]
            r["um_per_px_peakpicking_old"] = v_old
        except Exception:
            pass
        try:
            b = C.tick_blur_sigma(g)
            r.update(tick_sigma_px=b["sigma_px"], tick_width_px=b["tick_width_px"])
        except Exception as e:
            r["blur_error"] = str(e)
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(out / "ruler_frames.csv", index=False)
    ok = D.dropna(subset=["um_per_px"]) if "um_per_px" in D else D.iloc[:0]
    s = dict(pitch_um_assumed=pitch_um, n=int(len(ok)), method="tick-band autocorrelation (calibration.umpx_from_slide)")
    if len(ok):
        s["um_per_px"] = C.umpx_summary(ok.um_per_px.tolist())
        s["pitch_px_mean"] = float(ok.pitch_px.mean())
        s["long_tick_check"] = float(np.nanmedian(ok.long_over_10_fine)) if "long_over_10_fine" in ok else None
        if "um_per_px_peakpicking_old" in ok:
            s["um_per_px_old_peakpicking"] = float(ok.um_per_px_peakpicking_old.mean())
        if "tick_sigma_px" in ok:
            s["tick_sigma_px_median"] = float(ok.tick_sigma_px.median())
            s["tick_sigma_um_median"] = float(ok.tick_sigma_px.median() * s["um_per_px"]["mean"])
    return s


def run_blank(files, out):
    frames = [load_rgb(p)[..., 1].astype(float) for p in files]
    s = C.blank_card_noise(frames)
    w = C.white_from_pixel_noise(s["pixel_hp_noise_dn"], s["mean_dn"])
    fit = C.fit_corr_noise(s["band_dn"], w["white_dn"])
    s.update(white=w, corr_fit=fit)
    pd.DataFrame([dict(band_px=k, measured_dn=fit["measured_band_dn"][k], white_model_dn=fit["white_band_dn"][k],
                       fitted_dn=fit["fitted_band_dn"][k]) for k in fit["measured_band_dn"]]).to_csv(out / "noise_spectrum.csv", index=False)
    fig_noise(fit, out)
    return s


# ---------------------------------------------------------------- capillary zips
def group_tables(F, L):
    import analyze_session as AS
    U = AS.usable(F)
    LU = L[L.file.isin(U.file)]
    order = ["thumb", "pointer", "middle", "ring", "pinky"]
    fingers = [f for f in order if f in set(F.finger)] + sorted(set(F.finger) - set(order))
    rows = []
    for f in fingers:
        Fp, Uf, Lf = F[(F.finger == f) & F.primary], U[U.finger == f], LU[LU.finger == f]
        limbs = np.r_[Lf.limb_left.values, Lf.limb_right.values] if len(Lf) else np.array([])
        limbs = limbs[np.isfinite(limbs)]
        ap = Uf.median_apical_px.dropna()
        rows.append(dict(finger=f, primary_frames=int(len(Fp)), usable_frames=int(len(Uf)),
                         usable_pct=float(100 * len(Uf) / max(len(Fp), 1)),
                         loops_detected=int(L[(L.finger == f) & L.primary].shape[0]),
                         loops_resolved=int(np.isfinite(Lf.limb_left).sum()) if len(Lf) else 0,
                         limb_px_median=float(np.median(limbs)) if len(limbs) else np.nan,
                         apical_px_median=float(Lf.apical.median()) if len(Lf) else np.nan,
                         total_px_median=float(Lf.total.median()) if len(Lf) else np.nan,
                         sigma_px=float(Uf.sigma_px.median()) if len(Uf) else np.nan,
                         contrast=float(Uf.lumen_contrast_fit.median()) if len(Uf) else np.nan,
                         illum=float(Uf.illum_strength.median()) if len(Uf) else np.nan,
                         glare_pct=float(100 * Fp.glare_frac.median()) if len(Fp) else np.nan,
                         sharpness=float(Fp.sharpness.median()) if len(Fp) else np.nan,
                         gap_px=float(Uf.median_gap_px.median()) if len(Uf) else np.nan,
                         apical_cv_pct_between_captures=float(100 * ap.std(ddof=1) / ap.mean()) if len(ap) >= 2 else np.nan))
    per_finger = pd.DataFrame(rows)
    rows = []
    for k, g in F.groupby("kind"):
        rows.append(dict(condition=k, frames=int(len(g)), loops_per_frame=float(g.n_apices.mean()), resolved_per_frame=float(g.n_resolved.mean()),
                         sharpness=float(g.sharpness.median()), noise_dn=float(g.noise_dn.median()), glare_pct=float(100 * g.glare_frac.median()),
                         sigma_px=float(g.sigma_px.median()), apical_px=float(g.median_apical_px.median())))
    per_condition = pd.DataFrame(rows)
    F2 = F[F.primary].copy()
    rows = []
    for c, g in F2.groupby("capture"):
        u = g.file.isin(U.file)
        rows.append(dict(capture=int(c), primary_frames=int(len(g)), usable=int(u.sum()), usable_pct=float(100 * u.mean()),
                         glare_pct=float(100 * g.glare_frac.median()), sharpness=float(g.sharpness.median()),
                         resolved_per_frame=float(g.n_resolved.mean())))
    per_phase = pd.DataFrame(rows)
    rows = []
    for (who, hand), g in F2.groupby(["who", "hand"]) if "who" in F2 else []:
        u = U[(U.who == who) & (U.hand == hand)]
        lu = LU[LU.file.isin(u.file)]
        limbs = np.r_[lu.limb_left.values, lu.limb_right.values]
        limbs = limbs[np.isfinite(limbs)]
        rows.append(dict(person=who, hand=hand or "-", primary_frames=int(len(g)), usable=int(len(u)),
                         usable_pct=float(100 * len(u) / max(len(g), 1)), loops_resolved=int(np.isfinite(lu.limb_left).sum()),
                         limb_px_median=float(np.median(limbs)) if len(limbs) else np.nan,
                         apical_px_median=float(lu.apical.median()) if len(lu) else np.nan,
                         total_px_median=float(lu.total.median()) if len(lu) else np.nan,
                         gap_px=float(u.median_gap_px.median()) if len(u) else np.nan,
                         sigma_px=float(u.sigma_px.median()) if len(u) else np.nan,
                         glare_pct=float(100 * g.glare_frac.median())))
    per_person = pd.DataFrame(rows)
    return per_finger, per_condition, per_phase, per_person


def matched_frame(rng, px, sigma_px, contrast, illum, photons, corr_noise, texture, lumen_px=4.5, total_px=22.0, n=8):
    """A session-matched synthetic frame (pixel units), used to calibrate the texture estimator."""
    W, H = 640.0, 480.0
    specs = []
    for i in range(n):
        sp = G.make_loop(lumen_px, lumen_px + 1.5, total_px, limb_len=rng.uniform(90, 140))
        sp.cx = (i + 0.5) * W / n + rng.uniform(-0.15, 0.15) * W / n
        sp.y_base = H - rng.uniform(140, 260) - sp.limb_len
        specs.append(sp)
    mask = G.render_mask(specs, (W, H), 0.5)
    g = O.forward(mask, 0.5, sigma_um=sigma_px, contrast=contrast, illum_strength=illum, illum_angle_deg=rng.uniform(0, 360),
                  pixel_um=1.0, photons=photons, read_noise=0.0, exposure=0.6, rng=rng, texture_amp=texture,
                  texture_corr_um=25.0 / px, corr_noise=corr_noise)
    return (np.stack([np.clip(g * 1.45, 0, 1), g, np.clip(g * 0.62, 0, 1)], -1) * 255).round().astype(np.uint8)


def calibrate_texture(est_real, px, base, seed=0, levels=(0.0, 0.005, 0.01, 0.02, 0.04), reps=3):
    """Invert the texture estimator by simulation: estimator readings on matched frames with known texture,
    then interpolate the real median reading. Returns (texture_amp, table)."""
    rng = np.random.default_rng(seed)
    tab = []
    for t in levels:
        v = [RD.texture_strength(matched_frame(rng, px, base["sigma_px"], base["contrast"], base["illum"], base["photons"],
                                               base["corr_noise_frac"], t), 1.0, corr_um=25.0 / px) for _ in range(reps)]
        tab.append(dict(texture=t, est_mean=float(np.mean(v)), est_sd=float(np.std(v))))
    T = pd.DataFrame(tab)
    e = T.est_mean.values
    if est_real <= e[0]:
        return 0.0, T
    return float(np.interp(est_real, e, T.texture.values, right=np.nan)) if est_real <= e[-1] else float(T.texture.values[-1]), T


def build_scope(um_per_px, sigma_px, contrast, illum, noise, texture, notes):
    sc = ScopeParams(um_per_px=float(um_per_px), sigma_um=float(sigma_px * um_per_px), photons=float(noise["photons"]),
                     read_noise=0.0, illum_strength=float(illum), contrast=float(contrast),
                     corr_noise=noise["corr_noise_frac"], texture_amp=None if texture is None else float(texture))
    sc.measured.update(sigma_um=True, photons=noise["measured"], read_noise=noise["measured"], illum_strength=True, contrast=True)
    sc.notes = notes
    return sc


# ---------------------------------------------------------------- sims
def _run_one(args):
    name, scope_path, out, steps, quick = args
    cmd = [sys.executable, str(ROOT / "run_sims.py"), *steps, "--scope", str(scope_path), "--out", str(out),
           "--data", str(ROOT / "data")] + (["--quick"] if quick else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (Path(out) / "run.log").write_text(r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:])
    return name, r.returncode


def sim_metrics(out, sc):
    """Headline numbers from one scope's sim folders (one subfolder per step), at this scope's own blur and pixel size."""
    out = Path(out)
    m = {}
    S = {}
    for sj in sorted(out.glob("s*/summary.json")):
        S.update({k: v for k, v in json.loads(sj.read_text()).items() if k.startswith("s") and k[1:2].isdigit() or k == "texture_amp_used"})
    if not S:
        return dict(error="no sims run")
    f1 = out / "s1" / "s1_width_bias.csv"
    if f1.exists():
        d = pd.read_csv(f1)
        d = d[d.estimator == "fwhm"]
        sg = d.sigma.unique()[np.argmin(np.abs(d.sigma.unique() - sc.sigma_um))]
        pxv = d.pixel.unique()[np.argmin(np.abs(d.pixel.unique() - sc.um_per_px))]
        dd = d[(d.sigma == sg) & (d.pixel == pxv)]
        for k in ("limb", "apical", "total"):
            q = dd[dd.kind == k]
            if len(q):
                m[f"s1_{k}_mean_abs_bias_pct"] = float(q.bias_pct.abs().mean())
                m[f"s1_{k}_worst_bias_pct"] = float(q.bias_pct.iloc[q.bias_pct.abs().argmax()])
        m["s1_grid_sigma_um"], m["s1_grid_pixel_um"] = float(sg), float(pxv)
        m["s1_min_limb_under_10pct_um"] = S.get("s1_min_limb_under_10pct")
    f4 = out / "s4" / "s4_misclassification.csv"
    if f4.exists():
        d = pd.read_csv(f4)
        i = int(np.argmin(np.abs(d.sigma - sc.sigma_um)))
        m["s4_giant_error_at_scope_sigma"] = float(d.err_rate.iloc[i])
    if "s4_max_sigma_err_le_5pct" in S:
        m["s4_max_sigma_um_for_5pct"] = S["s4_max_sigma_err_le_5pct"].get("max_sigma_um")
    if "s4_density_call" in S:
        dc = S["s4_density_call"]
        m.update(s4_p7_called_reduced=dc["p_7_called_reduced"], s4_p5_called_normal=dc["p_5_called_normal"],
                 s4_density_est_9=dc["mean_est"].get("9"))
    if "s2_auc_reduced_density" in S:
        m["s2_auc_5_vs_7"] = {k: v.get("auc") for k, v in S["s2_auc_reduced_density"].items()}
    if "s7" in S:
        m.update(s7_icc=S["s7"].get("icc"), s7_cv_pct=S["s7"].get("cv_pct"))
    m["texture_used"] = S.get("texture_amp_used")
    return m


# ---------------------------------------------------------------- figures
def fig_noise(fit, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = list(fit["measured_band_dn"])
    x = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(x, [fit["measured_band_dn"][i] for i in k], "o-", color="#c0392b", label="measured (blank card, frame to frame)")
    ax.plot(x, [fit["white_band_dn"][i] for i in k], "s--", color="#7f8c8d", label="old sim model (white noise only)")
    ax.plot(x, [fit["fitted_band_dn"][i] for i in k], "^:", color="#2c6fbb", label="new sim model (white + correlated)")
    ax.set_xticks(x, [i + " px" for i in k])
    ax.set_yscale("log")
    ax.set_ylabel("noise, DN (one frame)")
    ax.set_xlabel("spatial scale band")
    ax.set_title("Camera noise by spatial scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_noise_spectrum.png", dpi=120)
    plt.close(fig)


def fig_fingers(PF, real_cv, sims, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    f = PF.finger.tolist()
    axs[0].bar(f, PF.usable_pct, color="#4a7ab5")
    for i, (u, n) in enumerate(zip(PF.usable_frames, PF.primary_frames)):
        axs[0].text(i, 2, f"{u}/{n}", ha="center", color="white", fontsize=9)
    axs[0].set_ylabel("usable raw frames (%)")
    axs[0].set_title("Usable captures per finger")
    axs[1].bar(f, PF.apical_px_median, color="#5a9e6f")
    axs[1].set_ylabel("median apical FWHM (px)")
    axs[1].set_title("Apical width per finger")
    sim_cv = [sims.get(f"finger_{x}", {}).get("s7_cv_pct", np.nan) for x in f]
    xx = np.arange(len(f))
    axs[2].bar(xx - 0.2, PF.apical_cv_pct_between_captures, 0.4, label="real (between captures)", color="#c0392b")
    axs[2].bar(xx + 0.2, sim_cv, 0.4, label="simulated (S7)", color="#7f8c8d")
    axs[2].set_xticks(xx, f, rotation=60 if len(f) > 6 else 0, fontsize=7 if len(f) > 6 else 9)
    for ax in axs[:2]:
        ax.tick_params(axis="x", rotation=60 if len(f) > 6 else 0, labelsize=7 if len(f) > 6 else 9)
    axs[2].set_ylabel("apical width CV (%)")
    axs[2].set_title("Repeatability: real vs model")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_per_finger.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main
def _numbers(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _numbers(v, f"{prefix}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _numbers(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)
    else:
        yield prefix, x


def compare_file(a, b):
    """'identical' | 'equal to 1e-9 (floating point)' | description of the real difference."""
    if sha256(a) == sha256(b):
        return "identical", 0.0
    if a.suffix == ".csv":
        A, B2 = pd.read_csv(a), pd.read_csv(b)
        if A.shape != B2.shape or list(A.columns) != list(B2.columns):
            return "different shape", float("inf")
        num = A.select_dtypes("number").columns
        if not (A.drop(columns=num).astype(str).values == B2.drop(columns=num).astype(str).values).all():
            return "text cells differ", float("inf")
        d = (A[num] - B2[num]).abs().values
        rel = d / (np.abs(A[num].values) + 1e-12)
        bad = (d > 1e-9) & (rel > 1e-9)
        if not bad.any():
            return "equal to 1e-9 (floating point)", float(np.nanmax(rel)) if rel.size else 0.0
        return f"{int(bad.sum())} of {bad.size} numbers differ, max |diff| {np.nanmax(d):.3g}", float(np.nanmax(d))
    if a.suffix == ".json":
        x, y = dict(_numbers(json.loads(a.read_text()))), dict(_numbers(json.loads(b.read_text())))
        if x.keys() != y.keys():
            return "different keys", float("inf")
        bad = [k for k in x if not (x[k] == y[k] or (isinstance(x[k], float) and isinstance(y[k], float)
                                                      and abs(x[k] - y[k]) <= 1e-9 * max(1.0, abs(x[k]))))]
        if not bad:
            return "equal to 1e-9 (floating point)", 0.0
        return f"{len(bad)} numbers differ ({', '.join(bad[:3])})", float("inf")
    return "bytes differ", float("inf")


def hash_outputs(d):
    res = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and p.suffix in HASHED and p.name not in NOT_HASHED:
            res[str(p.relative_to(d))] = sha256(p)
    return res


def analyse_capillary(r, o, work, noise, white_only, ruler):
    """Session analysis, tables, skin texture and scope files for one capillary input. Returns a JSON-able state."""
    import analyze_session as AS
    sess = work / (r["tag"] + "_capillary")
    if sess.exists():
        shutil.rmtree(sess)
    sess.mkdir()
    for p in r["_files"]:
        shutil.copyfile(p, sess / p.name)
    F, L, summ = AS.main(sess, o / "session")
    PF, PC, PP, PH = group_tables(F, L)
    PF.to_csv(o / "per_finger.csv", index=False)
    PC.to_csv(o / "per_condition.csv", index=False)
    PP.to_csv(o / "per_capture.csv", index=False)
    PH.to_csv(o / "per_person_hand.csv", index=False)
    U = AS.usable(F)
    if len(U) == 0:
        print(f"  {r['input']}: no usable frames, scopes and sims skipped")
        return dict(summary=summ, base=None, scopes={}, texture=dict(estimator_median=None, texture_amp=None, n_frames=0))
    base = dict(sigma_px=float(U.sigma_px.median()), contrast=float(U.lumen_contrast_fit.median()),
                illum=float(U.illum_strength.median()), photons=noise["photons"], corr_noise_frac=noise["corr_noise_frac"])
    px_c = SCALES["anat_central"]
    est = [RD.texture_strength(load_rgb(sess / f), px_c) for f in U.file]
    tex, ttab = calibrate_texture(float(np.median(est)), px_c, base)
    ttab.to_csv(o / "texture_calibration.csv", index=False)
    texture = dict(estimator_median=float(np.median(est)), texture_amp=tex, n_frames=len(est))
    common = dict(source=f"input sha256 {r['sha256'][:12]}", noise=noise["source"],
                  frames_measured=", ".join(f"{v} {k}" for k, v in F[F.primary].kind.value_counts().items()),
                  sigma_contrast_illum="median over usable measured frames (shared-sigma limb fit)",
                  texture=f"estimator {np.median(est):.4f} on {len(est)} frames, inverted by simulation -> {tex:.4f}")
    scopes = {}
    for nm, s_ in SCALES.items():
        scopes[nm] = build_scope(s_, base["sigma_px"], base["contrast"], base["illum"], noise, tex,
                                 dict(common, um_per_px=f"ANATOMICAL ESTIMATE {s_} (bounds 1.3-2.0), not the ruler"))
    scopes["white_noise_only"] = build_scope(px_c, base["sigma_px"], base["contrast"], base["illum"], white_only, tex,
                                             dict(common, ablation="correlated noise removed: what the sims assumed before the blank card"))
    for _, row in PF.iterrows():
        if row.usable_frames >= 1 and np.isfinite(row.sigma_px):
            scopes[f"finger_{row.finger}"] = build_scope(px_c, row.sigma_px, row.contrast, row.illum, noise, tex,
                                                         dict(common, finger=row.finger, usable_frames=int(row.usable_frames)))
    if ruler and "tick_sigma_px_median" in ruler:
        u = ruler["um_per_px"]["mean"]
        scopes["ruler_zoom"] = build_scope(u, ruler["tick_sigma_px_median"], base["contrast"], base["illum"], noise, tex,
                                           dict(common, um_per_px=f"RULER {u:.3f} um/px at the ruler's zoom; blur from the ruler ticks",
                                                caution="what nail captures WOULD give at the ruler's zoom; not these frames"))
        scopes["ruler_zoom"].measured["um_per_px"] = True
    paths = {}
    for nm, sc in scopes.items():
        p = o / "scopes" / f"{nm}.json"
        sc.save(p)
        paths[nm] = str(p)
    return dict(summary=summ, base=base, texture=texture, scopes=paths)


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _done(d, key):
    k = Path(d) / ".key"
    return k.exists() and k.read_text() == key


def _mark(d, key):
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / ".key").write_text(key)


class Budget:
    """Wall-clock budget so a run can be split across several calls (each unit is cached; rerun to continue)."""
    def __init__(self, seconds):
        import time
        self.t0, self.s, self.time = time.time(), seconds, time.time
        self.worked = False      # a call always does at least one unit, so every call makes progress

    def left(self):
        return float("inf") if self.s is None else self.s - (self.time() - self.t0)

    def allows(self, need):
        return not self.worked or self.left() >= need


def main(folder, out, sims="quick", jobs=4, pitch_um=10.0, keep_extracted=False, budget=None):
    folder, out = Path(folder).expanduser(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    B_ = Budget(budget)
    work = out / "_extracted"
    inputs = discover(folder)
    if not inputs:
        raise SystemExit(f"no zips or image folders in {folder}")
    print(f"{len(inputs)} inputs: {[p.name for p in inputs]}")
    chash, ahash, shash = code_hash(), analysis_hash(), sims_hash()
    runs, cal = [], dict(ruler=[], blank=[])
    seen, tags = {}, set()
    pending = 0
    for src in inputs:
        h = input_hash(src)
        base_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem if src.is_file() else src.name)
        tag, n = base_tag, 2
        while tag in tags:          # 'sujay copy.zip' and 'sujay_copy.zip' must not share an output folder
            tag, n = f"{base_tag}_{n}", n + 1
        tags.add(tag)
        o = out / tag
        k1 = _key("extract", h, ahash, pitch_um)
        r = dict(input=src.name, tag=tag, sha256=h, duplicate_of=seen.get(h), out=str(o))
        seen.setdefault(h, src.name)
        if _done(o, k1) and (work / tag).exists():
            st = json.loads((o / "input_state.json").read_text())
        else:
            if not B_.allows(20 + 0.5 * 60):
                pending += 1
                print(f"  {src.name}: deferred (time budget)")
                continue
            B_.worked = True
            d = extract(src, work / tag)
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            kinds = [(p, *classify_image(load_rgb(p))) for p in files]
            counts = pd.Series([k for _, k, _ in kinds]).value_counts().to_dict()
            kind = max(counts, key=counts.get) if counts else "empty"
            o.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(file=p.name, kind=k, **m) for p, k, m in kinds]).to_csv(o / "image_types.csv", index=False)
            st = dict(n_images=len(files), image_kinds=counts, kind=kind, files=[p.name for p, k, _ in kinds if k == kind])
            if kind == "ruler":
                st["result"] = run_ruler([d / f for f in st["files"]], o, pitch_um)
            elif kind == "blank":
                st["result"] = run_blank([d / f for f in st["files"]], o)
            (o / "input_state.json").write_text(json.dumps(st, indent=2, default=float))
            _mark(o, k1)
        r.update(n_images=st["n_images"], image_kinds=st["image_kinds"], kind=st["kind"], result=st.get("result"))
        r["_files"] = [work / tag / f for f in st["files"]]
        if r["kind"] in cal:
            cal[r["kind"]].append(r)
        runs.append(r)
        print(f"  {src.name}: {r['kind']} ({r['n_images']} images){'  [identical to ' + r['duplicate_of'] + ']' if r['duplicate_of'] else ''}", flush=True)

    # ---- shared calibration (first ruler zip, first blank zip; all are reported)
    ruler = cal["ruler"][0]["result"] if cal["ruler"] else None
    blank = cal["blank"][0]["result"] if cal["blank"] else None
    if blank:
        noise = dict(photons=blank["white"]["photons"], measured=True,
                     corr_noise_frac=[[s_, a_ / 255.0] for s_, a_ in blank["corr_fit"]["components"]],
                     source=f"blank card {cal['blank'][0]['input']}, {blank['n_frames']} frames at one exposure level")
    else:
        noise = dict(photons=50000.0, measured=False, corr_noise_frac=None, source="no blank-card zip: white-noise default")
    white_only = dict(noise, corr_noise_frac=None)

    # ---- capillary sessions
    import analyze_session as AS
    sim_jobs, scope_index = [], {}
    for r in runs:
        if r["kind"] != "capillary":
            continue
        o = out / r["tag"]
        k2 = _key("session", r["sha256"], ahash, noise, ruler)
        if _done(o / "session", k2):
            st = json.loads((o / "session_state.json").read_text())
        else:
            if not B_.allows(1.2 * len(r["_files"]) + 25):
                pending += 1
                print(f"  {r['input']}: session analysis deferred (time budget)", flush=True)
                continue
            B_.worked = True
            st = analyse_capillary(r, o, work, noise, white_only, ruler)
            (o / "session_state.json").write_text(json.dumps(st, indent=2, default=float))
            _mark(o / "session", k2)
        r.update(summary=st["summary"], base=st["base"], texture=st["texture"], scopes=st["scopes"])
        for nm, p in st["scopes"].items():
            sc = ScopeParams.load(p)
            scope_index[(r["tag"], nm)] = sc
            if sims != "none":
                for step in (FULL_STEPS if sims == "full" else QUICK_STEPS):     # one cached unit per scope and step
                    key = _key("sims", Path(p).read_text(), step, sims, shash)
                    d = o / "sims" / nm / step
                    if not _done(d, key):
                        sim_jobs.append((f"{r['tag']}|{nm}|{step}", p, d, [step], sims != "full", key))

    # ---- sims, in parallel, one process per scope, as many as the budget allows
    if sim_jobs:
        print(f"{len(sim_jobs)} sim steps ({sims}) to run on {jobs} workers", flush=True)
        sim_jobs.sort(key=lambda j: -STEP_COST.get(j[3][0], 60))     # longest first
        from concurrent.futures import FIRST_COMPLETED, wait
        todo, running, done_n = list(sim_jobs), {}, 0
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            while todo or running:
                while todo and len(running) < jobs:
                    nxt = next((j for j in todo if B_.allows(STEP_COST.get(j[3][0], 60) * (1 if sims != "full" else 6) + 8)), None)
                    if nxt is None:
                        break
                    todo.remove(nxt)
                    B_.worked = True
                    running[ex.submit(_run_one, nxt[:5])] = nxt
                if not running:
                    break
                fin, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in fin:
                    job = running.pop(f)
                    name, rc = f.result()
                    if rc == 0:
                        _mark(job[2], job[5])
                        done_n += 1
                    print(f"  {name}: {'ok' if rc == 0 else 'FAILED rc=' + str(rc)}", flush=True)
        i = len(sim_jobs) - len(todo)
        pending += len(sim_jobs) - i
    if pending:
        print(f"INCOMPLETE: {pending} units left. Run the same command again to continue (finished units are cached).")
        return None

    # ---- per-run results and manifests
    report_rows = []
    for r in runs:
        o = out / r["tag"]
        if r["kind"] == "capillary" and r.get("base"):
            sims_m = {}
            for nm in r["scopes"]:
                sims_m[nm] = sim_metrics(o / "sims" / nm, scope_index[(r["tag"], nm)]) if sims != "none" else {}
            (o / "sim_metrics.json").write_text(json.dumps(sims_m, indent=2, default=float))
            pd.DataFrame([dict(scenario=k, **v) for k, v in sims_m.items()]).to_csv(o / "sim_metrics.csv", index=False)
            PF = pd.read_csv(o / "per_finger.csv")
            fig_fingers(PF, None, sims_m, o)
            r["sims"] = sims_m
        man = dict(input=r["input"], sha256=r["sha256"], kind=r["kind"], n_images=r["n_images"], image_kinds=r["image_kinds"],
                   code_sha256=code_hash(), versions=versions(), sims=sims, pitch_um=pitch_um, outputs=hash_outputs(o))
        (o / "manifest.json").write_text(json.dumps(man, indent=2, default=float))
        r["outputs"] = man["outputs"]
        report_rows.append(r)

    # ---- the same capture seen as raw in one zip and enhanced in another (or the same zip)
    pair_rows = []
    frames = []
    for r in runs:
        fp = out / r["tag"] / "session" / "frames.csv"
        if r["kind"] == "capillary" and fp.exists():
            f = pd.read_csv(fp)
            f["zip"] = r["input"]
            f["stem"] = f.file.str.replace(r"-?[neg]\.png$", "", regex=True)
            frames.append(f)
    if frames:
        A = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["file"])
        for stem, g in A.groupby("stem"):
            rr, ee = g[g.kind == "raw"], g[g.kind == "enhanced"]
            how = "pixels"
            if not (len(rr) and len(ee)):          # same colours (e.g. a sharpening-only 'enhanced' view): use the file labels
                rr, ee, how = g[g.suffix == "n"], g[g.suffix == "e"], "file label"
            if len(rr) and len(ee):
                a, b = rr.iloc[0], ee.iloc[0]
                pair_rows.append(dict(capture=stem, raw_file=a.file, enh_file=b.file, paired_by=how, zip_raw=a.zip, zip_enh=b.zip,
                                      **{f"raw_{k}": a[k] for k in ("n_apices", "n_resolved", "sharpness", "noise_dn", "sigma_px", "median_apical_px", "glare_frac")},
                                      **{f"enh_{k}": b[k] for k in ("n_apices", "n_resolved", "sharpness", "noise_dn", "sigma_px", "median_apical_px", "glare_frac")}))
    P = pd.DataFrame(pair_rows)
    paired = {}
    if len(P):
        P.to_csv(out / "paired_raw_vs_enhanced.csv", index=False)
    for grp_name, PP_ in ([("all", P)] + [(f"{zr} vs {ze}", g) for (zr, ze), g in P.groupby(["zip_raw", "zip_enh"])]) if len(P) else []:
        paired[grp_name] = {}
        P_ = PP_
        for k in ("n_apices", "n_resolved", "sharpness", "noise_dn", "sigma_px", "median_apical_px"):
            Q = P_[[f"raw_{k}", f"enh_{k}"]].dropna()
            d = Q[f"enh_{k}"] - Q[f"raw_{k}"]
            paired[grp_name][k] = dict(n=int(len(d)), raw_mean=float(Q[f"raw_{k}"].mean()) if len(Q) else None, enh_mean=float(Q[f"enh_{k}"].mean()) if len(Q) else None,
                             mean_diff=float(d.mean()) if len(d) else None,
                             p_wilcoxon=float(__import__("scipy").stats.wilcoxon(d).pvalue) if len(d) >= 6 and d.abs().sum() > 0 else None)

    # ---- determinism across identical inputs
    det = []
    by_hash = {}
    for r in runs:
        by_hash.setdefault(r["sha256"], []).append(r)
    for h, group in by_hash.items():
        if len(group) > 1:
            a = group[0]
            for b in group[1:]:
                diff = sorted(k for k in set(a["outputs"]) | set(b["outputs"]) if a["outputs"].get(k) != b["outputs"].get(k))
                verdicts = {k: compare_file(out / a["tag"] / k, out / b["tag"] / k) if (out / a["tag"] / k).exists() and (out / b["tag"] / k).exists()
                            else ("missing", float("inf")) for k in diff}
                fp = [k for k, v in verdicts.items() if v[0].startswith("equal")]
                real = {k: v[0] for k, v in verdicts.items() if not v[0].startswith("equal")}
                det.append(dict(a=a["input"], b=b["input"], sha256=h[:12], files_compared=len(a["outputs"]),
                                byte_identical=len(a["outputs"]) - len(diff), float_equal=len(fp), real_differences=real,
                                identical=not real))
    other = []
    for f in sorted(out.glob("cross_machine*.json")):
        other += [r["unit"] for r in json.loads(f.read_text()) if r.get("action") == "filled"]
    batch = dict(folder=str(folder), sims=sims, code_sha256=code_hash(), versions=versions(),
                 inputs=[{k: v for k, v in r.items() if not k.startswith("_") and k not in ("outputs", "summary")} for r in runs],
                 calibration=dict(ruler=ruler, blank=blank, noise_used=noise), determinism=det, paired_raw_vs_enhanced=paired,
                 units_from_other_machine=len(other))
    (out / "batch_summary.json").write_text(json.dumps(batch, indent=2, default=float))
    write_report(out, batch, runs)
    if not keep_extracted:
        shutil.rmtree(work, ignore_errors=True)
    print("->", out)
    return batch


def _f(x, nd=2):
    try:
        return "n/a" if x is None or not np.isfinite(float(x)) else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def write_report(out, batch, runs):
    L = ["# Batch run", "", f"Code sha256 `{batch['code_sha256'][:12]}`, sims: {batch['sims']}, "
         f"python {batch['versions']['python']}, numpy {batch['versions']['numpy']}.", "", "## Inputs", "",
         "| input | sha256 | images | type | note |", "|---|---|---|---|---|"]
    for r in batch["inputs"]:
        L.append(f"| {r['input']} | `{r['sha256'][:12]}` | {r['n_images']} | {r['kind']} | {'identical to ' + r['duplicate_of'] if r['duplicate_of'] else ''} |")
    c = batch["calibration"]
    L += ["", "## Calibration", ""]
    if c["ruler"]:
        u = c["ruler"]["um_per_px"]
        L.append(f"- Ruler: **{_f(u['mean'], 3)} um/px** (SD {_f(u['sd'], 3)}, n={c['ruler']['n']}) if one fine tick = "
                 f"{c['ruler']['pitch_um_assumed']} um; blur at the ruler's zoom {_f(c['ruler'].get('tick_sigma_px_median'))} px.")
    if c["blank"]:
        b = c["blank"]
        L.append(f"- Blank card: {b['n_frames']} frames at ~{_f(b['mean_dn'], 0)} DN; per-pixel noise {_f(b['pixel_hp_noise_dn'])} DN; "
                 f"brightness drift {_f(b['drift_dn'], 1)} DN across frames; flat field centre/edge {_f(b['flat_centre_over_edge'], 3)}.")
        L.append("- Noise by scale (DN): " + ", ".join(f"{k} px {_f(v)}" for k, v in b["band_dn"].items()))
    L.append(f"- Noise model used in the sims: {c['noise_used']['source']}.")
    for r in runs:
        if r["kind"] != "capillary":
            continue
        o = out / r["tag"]
        L += ["", f"## {r['input']}", ""]
        s = r["summary"]
        kinds = ", ".join(f"{v} {k}" for k, v in s["kinds"].items())
        L.append(f"{s['n_files']} files ({kinds} by pixels), {s['n_primary']} measured frames (one per capture, raw preferred), {s['n_usable']} usable, "
                 f"{s['loops_resolved_usable']} loops with both limbs. Skin texture {_f(r['texture']['texture_amp'], 4)} (calibrated).")
        PF = pd.read_csv(o / "per_finger.csv")
        L += ["", "| finger | usable | loops | limb px | apical px | total px | blur px | glare % | apical CV % (real) | S7 CV % (sim) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, x in PF.iterrows():
            sm = r["sims"].get(f"finger_{x.finger}", {}) if r.get("sims") else {}
            L.append(f"| {x.finger} | {int(x.usable_frames)}/{int(x.primary_frames)} | {int(x.loops_resolved)} | {_f(x.limb_px_median, 1)} | "
                     f"{_f(x.apical_px_median, 1)} | {_f(x.total_px_median, 1)} | {_f(x.sigma_px)} | {_f(x.glare_pct, 1)} | "
                     f"{_f(x.apical_cv_pct_between_captures, 1)} | {_f(sm.get('s7_cv_pct'), 1)} |")
        PH = pd.read_csv(o / "per_person_hand.csv") if (o / "per_person_hand.csv").exists() else pd.DataFrame()
        if len(PH) > 1:
            L += ["", "| person | hand | usable | loops | limb px | apical px | total px | gap px | blur px |", "|---|---|---|---|---|---|---|---|---|"]
            for _, x in PH.iterrows():
                L.append(f"| {x.person} | {x.hand} | {int(x.usable)}/{int(x.primary_frames)} | {int(x.loops_resolved)} | {_f(x.limb_px_median, 1)} | "
                         f"{_f(x.apical_px_median, 1)} | {_f(x.total_px_median, 1)} | {_f(x.gap_px, 0)} | {_f(x.sigma_px)} |")
        if r.get("sims"):
            L += ["", "| scenario | um/px | limb bias % | apical bias % | total bias % | 45 vs 55 error | 7/mm called reduced | S7 ICC | S7 CV % |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for nm, m in r["sims"].items():
                sc = ScopeParams.load(r["scopes"][nm])
                L.append(f"| {nm} | {_f(sc.um_per_px, 3)} | {_f(m.get('s1_limb_mean_abs_bias_pct'), 1)} | {_f(m.get('s1_apical_mean_abs_bias_pct'), 1)} | "
                         f"{_f(m.get('s1_total_mean_abs_bias_pct'), 1)} | {_f(m.get('s4_giant_error_at_scope_sigma'), 3)} | "
                         f"{_f(m.get('s4_p7_called_reduced'), 2)} | {_f(m.get('s7_icc'), 3)} | {_f(m.get('s7_cv_pct'), 1)} |")
    if batch.get("paired_raw_vs_enhanced"):
        L += ["", "## Same capture, raw vs enhanced"]
        for grp, res in batch["paired_raw_vs_enhanced"].items():
            L += ["", f"**{grp}**", "", "| measure | pairs | raw | enhanced | mean diff (enh - raw) | Wilcoxon p |", "|---|---|---|---|---|---|"]
            for k, v in res.items():
                L.append(f"| {k} | {v['n']} | {_f(v['raw_mean'])} | {_f(v['enh_mean'])} | {_f(v['mean_diff'])} | {_f(v['p_wilcoxon'], 4)} |")
    if batch["determinism"]:
        L += ["", "## Reproducibility (identical inputs)", "",
              f"{batch.get('units_from_other_machine', 0)} sim units were computed on a second machine (different numpy/scipy) and merged; "
              "differences are classified as byte-identical, equal to 1e-9 (floating point) or real.", "",
              "| a | b | output files | byte-identical | equal to 1e-9 | real differences |", "|---|---|---|---|---|---|"]
        for d in batch["determinism"]:
            rd = "; ".join(f"{k}: {v}" for k, v in d["real_differences"].items()) or "none"
            L.append(f"| {d['a']} | {d['b']} | {d['files_compared']} | {d['byte_identical']} | {d['float_equal']} | {rd} |")
    L += ["", "All widths are model-checked but NOT validated against human measurement (S5). um values use the anatomical",
          "scale unless the scenario says ruler. Reproduce: `python batch_run.py <folder> --out <out>`."]
    (out / "BATCH_REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/batch")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true", help="full-length sims (slow)")
    g.add_argument("--sims", choices=["quick", "full", "none"], default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--pitch-um", type=float, default=10.0, help="ruler fine-tick spacing in um")
    ap.add_argument("--keep-extracted", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="stop after about this many seconds; rerun to continue (cached)")
    a = ap.parse_args()
    res = main(a.folder, a.out, sims=a.sims or ("full" if a.full else "quick"), jobs=a.jobs, pitch_um=a.pitch_um,
               keep_extracted=a.keep_extracted, budget=a.budget)
    sys.exit(3 if res is None else 0)
