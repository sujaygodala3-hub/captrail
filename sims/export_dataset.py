"""Turn a batch run (batch_run.py output) into a versioned, documented training dataset.

    python export_dataset.py results/batch --out datasets/captrail-v3
    python export_dataset.py results/batch --out datasets/captrail-v3 --compare-with <second batch dir>
    python export_dataset.py --check datasets/captrail-v3          # verify hashes, schema, splits, privacy
    python export_dataset.py results/batch --out datasets/captrail-v3 --vision-images <folder of zips> \
        --vision-out datasets/_local_vision                         # images + COCO keypoints, stays LOCAL

What it writes (all text, no images, subjects pseudonymized):
    tables/      sessions, frames, loops, sites, paired_views, splits          (one row per thing, tidy)
    calibration/ every card/ruler photo (nail zoom or not), the scale used, blank-card noise spectrum
    sims/        scenarios (every scope the sims ran), headline metrics, and the S1 S2 S4 S6 S7 curves
    qa/          reproducibility on identical inputs, optional cross-platform stability
    llm/         records.jsonl (one per frame) and chat-format SFT files per split
    README.md    dataset card;  SCHEMA.md  every column with type, unit and meaning;  manifest.json  hashes

Deterministic: same batch in, byte-identical dataset out (no timestamps, sorted rows, fixed rounding).
Identical input zips are counted once. Real subject names and original file names go only to
<out>/_private/ (git-ignored). Every label here is produced by the pipeline, not by a clinician.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VERSION = "3.0.0"   # 3: measured nail-zoom scale (*_um columns), card photos from every zoom, calibration_qc task
SIG = 6                                   # significant digits kept in every float
UM_PER_PX_EST, UM_LOW, UM_HIGH = 1.6, 1.3, 2.0   # anatomical fallback, used only when the batch has no nail-zoom card
STEPS = ("s1", "s2", "s4", "s6", "s7")
SPLIT_FRACTIONS = (("test", 0.15), ("val", 0.15))   # the rest is train
SYSTEM_BASE = ("You are the Captrail measurement assistant for a low-cost USB nailfold capillaroscope. "
               "You judge image quality, report capillary measurements and explain how reliable they are. "
               "Captrail is a student research prototype, not a diagnostic device: never diagnose. ")
SCALE = {}       # set by build(): the scale every um column uses (see scale_of)


def scale_of(bs):
    """The scale for every um value: the card photographed at nail zoom when the batch has one (status
    'measured'), else the anatomical estimate. low/high: mean -/+ 2 SD over the card photos (measured) or the
    anatomical bounds."""
    sc = (bs.get("calibration") or {}).get("scale") or {}
    if sc.get("status") == "measured":
        u, sd = float(sc["um_per_px"]), float(sc.get("sd") or 0.0)
        return dict(um=u, low=u - 2 * sd, high=u + 2 * sd, status="measured", n=sc.get("n"), source=sc.get("source", ""),
                    tick_sigma_px=sc.get("tick_sigma_px"), aspect=sc.get("aspect_v_over_h"))
    return dict(um=UM_PER_PX_EST, low=UM_LOW, high=UM_HIGH, status="anatomical_estimate", n=0,
                source="anatomical estimate from healthy loop spacing and widths (no card at nail zoom)")


def system_prompt():
    if SCALE.get("status") == "measured":
        return SYSTEM_BASE + (f"Widths are measured in pixels and converted with the scale measured from the calibration card at the "
                              f"nail zoom ({SCALE['um']:.4f} um per pixel). If zoom, height or focus changes, the card must be photographed again.")
    return SYSTEM_BASE + ("Widths are measured in pixels; micrometre values use an estimated scale until the card is "
                          "photographed at the same zoom as the nail.")

# ------------------------------------------------------------------------------------------- schema
# (column, type, unit, meaning). Exports select exactly these columns, in this order; --check enforces it.
S = {}
S["sessions"] = [
    ("session_id", "str", "", "Capture session, S01, S02, ..."),
    ("subjects", "str", "", "Pseudonymized subjects in the session, ';'-separated"),
    ("views", "str", "", "Views present: raw (unprocessed camera frame), enhanced (app-processed), grey"),
    ("n_inputs", "int", "", "Distinct input zips (identical copies counted once)"),
    ("n_duplicate_inputs", "int", "", "Byte-identical copies of an input that were ignored"),
    ("input_sha256", "str", "", "sha256 of each distinct input zip, ';'-separated"),
    ("n_frames", "int", "", "Image files in the session"),
    ("n_measured_frames", "int", "", "Measured frames: one per capture per input zip (raw preferred)"),
    ("n_usable_frames", "int", "", "Measured frames passing the usability rule"),
    ("n_loops_resolved", "int", "", "Loops with both limbs measured, usable frames"),
    ("um_per_px", "float", "um/px", "Scale used for every *_um column"),
    ("um_per_px_low", "float", "um/px", "Lower bound: mean - 2 SD over the card photos (measured) or the anatomical bound"),
    ("um_per_px_high", "float", "um/px", "Upper bound: mean + 2 SD over the card photos (measured) or the anatomical bound"),
    ("scale_status", "str", "", "measured (calibration card photographed at nail zoom) or anatomical_estimate"),
    ("scale_source", "str", "", "Where the scale comes from"),
]
S["frames"] = [
    ("frame_id", "str", "", "Stable id: session/subject-hand+digit-capture-view"),
    ("session_id", "str", "", "Session"),
    ("subject_id", "str", "", "Pseudonymized subject"),
    ("site_id", "str", "", "Subject + hand + digit; the unit for train/val/test splits"),
    ("hand", "str", "", "L, R or empty when not recorded"),
    ("digit", "str", "", "thumb, pointer, middle, ring, pinky"),
    ("capture", "int", "", "Capture number at this site (1 = first)"),
    ("view", "str", "", "From the file label: raw, enhanced or grey"),
    ("view_by_pixels", "str", "", "What the pixel classifier says: raw, enhanced, grey"),
    ("view_label_matches_pixels", "bool", "", "False when the file label and the pixels disagree"),
    ("measured", "bool", "", "This frame is the one measured for its capture within its input zip (raw preferred; an enhanced-only zip measures its enhanced frames)"),
    ("usable", "bool", "", "measured AND >= 3 loops with both limbs AND glare < 30%"),
    ("unusable_reason", "str", "", "Why not usable: not_measured_view, too_few_resolved_loops, glare (';'-separated)"),
    ("split", "str", "", "train, val or test (by site)"),
    ("n_apices", "int", "count", "Loop apexes detected"),
    ("n_loops_measured", "int", "count", "Loops measured (away from edges and glare)"),
    ("n_loops_resolved", "int", "count", "Loops with both limbs resolved"),
    ("glare_pct", "float", "%", "Frame area saturated by LED glare"),
    ("clipped_pct", "float", "%", "Pixels at >= 250 in any channel"),
    ("sharpness", "float", "DN^2", "Variance of the Laplacian outside glare"),
    ("noise_dn", "float", "DN", "Robust high-pass noise of the green channel"),
    ("mean_r", "float", "DN", "Mean red outside glare"),
    ("mean_g", "float", "DN", "Mean green outside glare"),
    ("mean_b", "float", "DN", "Mean blue outside glare"),
    ("illum_strength", "float", "fraction", "Peak-to-peak lighting gradient across the frame"),
    ("blur_sigma_px", "float", "px", "Gaussian blur from the shared-sigma limb fit (needs >= 4 limbs)"),
    ("lumen_width_fit_px", "float", "px", "Median lumen width from that fit, blur removed"),
    ("lumen_contrast", "float", "fraction", "Median fractional darkening of the limbs"),
    ("median_gap_px", "float", "px", "Median spacing between neighbouring apexes"),
    ("median_limb_left_px", "float", "px", "Median left (arterial-side) limb FWHM"),
    ("median_limb_right_px", "float", "px", "Median right (venous-side) limb FWHM"),
    ("median_apical_px", "float", "px", "Median apical FWHM"),
    ("median_total_px", "float", "px", "Median total loop width, outer edge to outer edge"),
    ("median_apical_um", "float", "um", "median_apical_px x um_per_px (sessions.csv)"),
    ("median_total_um", "float", "um", "median_total_px x um_per_px"),
]
S["loops"] = [
    ("loop_id", "str", "", "frame_id + '/L' + index left to right"),
    ("frame_id", "str", "", "Frame"),
    ("session_id", "str", "", "Session"),
    ("site_id", "str", "", "Site"),
    ("view", "str", "", "raw, enhanced or grey"),
    ("frame_usable", "bool", "", "The frame passed the usability rule"),
    ("split", "str", "", "train, val or test"),
    ("apex_x_px", "float", "px", "Apex column in the 640 x 480 frame"),
    ("apex_y_px", "float", "px", "Apex row"),
    ("resolved", "bool", "", "Both limbs measured"),
    ("limb_left_px", "float", "px", "Left limb FWHM"),
    ("limb_right_px", "float", "px", "Right limb FWHM"),
    ("apical_px", "float", "px", "Apical FWHM"),
    ("total_px", "float", "px", "Total loop width"),
    ("depth_frac", "float", "fraction", "Darkening of the limbs relative to background"),
    ("limb_left_um", "float", "um", "limb_left_px x um_per_px"),
    ("limb_right_um", "float", "um", "limb_right_px x um_per_px"),
    ("apical_um", "float", "um", "apical_px x um_per_px"),
    ("total_um", "float", "um", "total_px x um_per_px"),
]
S["sites"] = [
    ("site_id", "str", "", "Subject + hand + digit"),
    ("session_id", "str", "", "Session"),
    ("view", "str", "", "View the measurements come from"),
    ("split", "str", "", "train, val or test"),
    ("measured_frames", "int", "", "Measured frames (one per capture)"),
    ("usable_frames", "int", "", "Usable frames"),
    ("usable_pct", "float", "%", "usable / measured"),
    ("loops_detected", "int", "", "Loops detected on measured frames"),
    ("loops_resolved", "int", "", "Loops resolved on usable frames"),
    ("limb_px_median", "float", "px", "Median limb FWHM"),
    ("apical_px_median", "float", "px", "Median apical FWHM"),
    ("total_px_median", "float", "px", "Median total width"),
    ("blur_sigma_px", "float", "px", "Median blur"),
    ("lumen_contrast", "float", "fraction", "Median limb contrast"),
    ("illum_strength", "float", "fraction", "Median lighting gradient"),
    ("glare_pct", "float", "%", "Median glare"),
    ("sharpness", "float", "DN^2", "Median sharpness"),
    ("gap_px", "float", "px", "Median apex spacing"),
    ("apical_cv_pct_between_captures", "float", "%", "CV of per-capture median apical width (real repeatability)"),
]
S["paired_views"] = [
    ("capture_id", "str", "", "Session/subject-hand+digit-capture, seen in both views"),
    ("session_id", "str", "", "Session"),
    ("site_id", "str", "", "Site"),
    ("split", "str", "", "train, val or test"),
    ("paired_by", "str", "", "pixels or file label"),
] + [(f"{v}_{k}", t, u, f"{d} ({v} view)") for v in ("raw", "enhanced") for k, t, u, d in (
    ("n_apices", "int", "count", "Apexes detected"), ("n_loops_resolved", "int", "count", "Loops resolved"),
    ("sharpness", "float", "DN^2", "Sharpness"), ("noise_dn", "float", "DN", "Noise"),
    ("blur_sigma_px", "float", "px", "Blur"), ("median_apical_px", "float", "px", "Median apical FWHM"),
    ("glare_pct", "float", "%", "Glare"))]
S["splits"] = [
    ("site_id", "str", "", "Site"),
    ("session_id", "str", "", "Session"),
    ("split", "str", "", "train, val or test"),
    ("n_frames", "int", "", "Frames at this site (all views)"),
    ("rank_key", "str", "", "First 12 hex of sha256(site_id); sites are dealt to splits in this order"),
]
S["ruler"] = [
    ("input_id", "str", "", "Input alias (IN-xx) the photo came from"),
    ("photo", "str", "", "Photo number within that input"),
    ("nail_zoom", "bool", "", "True when this card photo is at the nail zoom (its scale is the one every um column uses)"),
    ("method", "str", "", "crosshair tick ladder (high zoom) or tick-band autocorrelation (low zoom)"),
    ("um_per_px", "float", "um/px", "Scale at this photo's zoom (one fine tick = 100 um)"),
    ("se_um_per_px", "float", "um/px", "Standard error of that photo's fit"),
    ("pitch_px", "float", "px", "Fine-tick pitch"),
    ("aspect_v_over_h", "float", "ratio", "Vertical-arm pitch / horizontal-arm pitch (1.0 = square pixels); ladder only"),
    ("tilt_deg", "float", "deg", "Best rotation"),
    ("long_tick_px", "float", "px", "Long-tick spacing (independent check, 10 pitches expected)"),
    ("long_over_10_fine", "float", "ratio", "long_tick_px / (10 x pitch_px); 1.0 = consistent"),
    ("um_per_px_old_peakpicking", "float", "um/px", "Superseded estimator (split ticks), kept for the record"),
    ("tick_blur_sigma_px", "float", "px", "Blur at the ruler's zoom, from folded tick profile"),
    ("tick_width_px", "float", "px", "Printed tick width"),
]
S["noise_spectrum"] = [
    ("band_px", "str", "px", "Difference-of-Gaussian band"),
    ("measured_dn", "float", "DN", "Blank-card frame-to-frame noise in this band (one frame)"),
    ("white_model_dn", "float", "DN", "What a white-noise model gives"),
    ("fitted_dn", "float", "DN", "White + correlated model used in the sims"),
]
S["scenarios"] = [
    ("scenario_id", "str", "", "session-view-scenario"),
    ("session_id", "str", "", "Session whose frames set the parameters"),
    ("view", "str", "", "View the parameters were measured on"),
    ("scenario", "str", "", "measured (the measured nail-zoom scale) or anat_central / anat_low / anat_high (no card at nail zoom), white_noise_only (ablation), site_* (one site), ruler_zoom (what-if at the low-zoom card)"),
    ("site_id", "str", "", "Site for site_* scenarios, else empty"),
    ("um_per_px", "float", "um/px", "Pixel size simulated"),
    ("sigma_um", "float", "um", "Blur simulated"),
    ("contrast", "float", "fraction", "Lumen contrast"),
    ("illum_strength", "float", "fraction", "Lighting gradient"),
    ("photons", "float", "", "Photons per pixel at full scale (white part of the noise)"),
    ("corr_noise", "str", "", "JSON [[scale_px, std], ...] correlated noise, empty = white only"),
    ("texture_amp", "float", "fraction", "Skin texture amplitude used in S2/S4"),
    ("scale_status", "str", "", "measured, anatomical_estimate, or ruler (ruler_zoom: the low-zoom card, a what-if)"),
]
S["metrics"] = [
    ("scenario_id", "str", "", "Scenario"),
    ("mode", "str", "", "quick (few replicates) or full"),
    ("s1_limb_mean_abs_bias_pct", "float", "%", "Mean |bias| of limb FWHM over the S1 truth grid at this blur and pixel"),
    ("s1_apical_mean_abs_bias_pct", "float", "%", "Same for apical width"),
    ("s1_total_mean_abs_bias_pct", "float", "%", "Same for total width"),
    ("s1_limb_worst_bias_pct", "float", "%", "Signed bias of the worst limb case"),
    ("s1_apical_worst_bias_pct", "float", "%", "Signed bias of the worst apical case"),
    ("s1_total_worst_bias_pct", "float", "%", "Signed bias of the worst total case"),
    ("s1_min_limb_under_10pct_um", "float", "um", "Smallest true limb measured within 10%"),
    ("s4_giant_error", "float", "fraction", "P(45 um read as giant or 55 um read as not), averaged"),
    ("s4_max_sigma_um_for_5pct", "float", "um", "Largest blur keeping that error <= 5%"),
    ("s4_p7_called_reduced", "float", "fraction", "P(true 7/mm called < 7/mm)"),
    ("s4_p5_called_normal", "float", "fraction", "P(true 5/mm called >= 7/mm)"),
    ("s4_density_est_9", "float", "loops/mm", "Mean estimate for a true 9/mm"),
    ("s7_icc", "float", "", "Simulated repeat-capture ICC(2,1)"),
    ("s7_cv_pct", "float", "%", "Simulated repeat-capture CV of apical width"),
    ("platform_unstable", "str", "", "Metrics that changed between two platforms (';'-separated); empty = stable or not checked"),
]
S["s1_width_bias"] = [("scenario_id", "str", "", "Scenario"), ("kind", "str", "", "limb, apical or total"),
                      ("true_um", "float", "um", "True width"), ("sigma_um", "float", "um", "Blur"), ("pixel_um", "float", "um", "Pixel"),
                      ("estimator", "str", "", "fwhm (half-depth) or fixed threshold"), ("mean_um", "float", "um", "Mean measured"),
                      ("sd_um", "float", "um", "SD measured"), ("bias_pct", "float", "%", "Mean bias"), ("n_ok", "int", "", "Replicates measured"),
                      ("bias_lo95", "float", "%", "95% CI low"), ("bias_hi95", "float", "%", "95% CI high")]
S["s2_density"] = [("scenario_id", "str", "", "Scenario"), ("field_mm", "float", "mm", "Field width"), ("sigma_um", "float", "um", "Blur"),
                   ("contrast", "float", "fraction", "Contrast"), ("true_density", "float", "loops/mm", "Truth"), ("rep", "int", "", "Replicate"),
                   ("est_density", "float", "loops/mm", "Estimate"), ("err_pct", "float", "%", "Error")]
S["s2_texture"] = [("scenario_id", "str", "", "Scenario"), ("texture", "float", "fraction", "Texture amplitude"),
                   ("true_density", "float", "loops/mm", "Truth"), ("rep", "int", "", "Replicate"), ("est_density", "float", "loops/mm", "Estimate")]
S["s4_misclassification"] = [("scenario_id", "str", "", "Scenario"), ("sigma_um", "float", "um", "Blur"),
                             ("p_45_called_giant", "float", "fraction", "P(45 um read >= 50)"), ("p_55_called_not_giant", "float", "fraction", "P(55 um read < 50)"),
                             ("p45_lo95", "float", "fraction", "CI"), ("p45_hi95", "float", "fraction", "CI"), ("p55_lo95", "float", "fraction", "CI"),
                             ("p55_hi95", "float", "fraction", "CI"), ("n_each", "int", "", "Trials"), ("err_rate", "float", "fraction", "Mean of the two"),
                             ("mean_45", "float", "um", "Mean read for 45"), ("mean_55", "float", "um", "Mean read for 55")]
S["s6_propagation"] = [("scenario_id", "str", "", "Scenario"), ("true_width_um", "float", "um", "True width"),
                       ("assumption", "str", "", "resistance, flow, velocity, pressure, ..."), ("exponent", "float", "", "Power of radius"),
                       ("mean_pct", "float", "%", "Mean error after propagation"), ("sd_pct", "float", "%", "SD"), ("p5_pct", "float", "%", "5th pct"),
                       ("median_pct", "float", "%", "Median"), ("p95_pct", "float", "%", "95th pct")]
S["s7_repeatability"] = [("scenario_id", "str", "", "Scenario"), ("icc", "float", "", "ICC(2,1)"), ("ci_lo", "float", "", "Bootstrap 95% low"),
                         ("ci_hi", "float", "", "Bootstrap 95% high"), ("icc_oracle_position", "float", "", "ICC if the tool knew the true position"),
                         ("cv_pct", "float", "%", "Repeat CV"), ("n_subjects", "int", "", "Simulated subjects")]
S["reproducibility"] = [("input_a", "str", "", "Input zip"), ("input_b", "str", "", "Byte-identical input zip"),
                        ("output_files", "int", "", "Outputs compared"), ("byte_identical", "int", "", "Identical bytes"),
                        ("float_equal", "int", "", "Equal to 1e-9"), ("real_differences", "str", "", "Anything else")]
S["platform_stability"] = [("metric", "str", "", "Headline sim metric"), ("n_scenarios", "int", "", "Scenarios compared"),
                           ("n_differ", "int", "", "Scenarios where it changed"), ("max_abs_diff", "float", "", "Largest change"),
                           ("median_abs_diff", "float", "", "Median change")]

ROW_DESC = {"sessions": "capture session", "frames": "image file", "loops": "detected capillary loop",
            "sites": "finger site and view", "paired_views": "capture seen in both raw and enhanced view",
            "splits": "finger site and its split", "ruler": "calibration card photo", "noise_spectrum": "noise band of the blank card",
            "scenarios": "simulated scope (parameters)", "metrics": "simulated scope (headline errors)",
            "s1_width_bias": "S1 width point", "s2_density": "S2 density trial", "s2_texture": "S2 texture trial",
            "s4_misclassification": "S4 blur level", "s6_propagation": "S6 width and assumption",
            "s7_repeatability": "S7 scenario", "reproducibility": "pair of identical inputs",
            "platform_stability": "headline metric across two platforms"}

TABLE_PATH = {"sessions": "tables/sessions.csv", "frames": "tables/frames.csv", "loops": "tables/loops.csv",
              "sites": "tables/sites.csv", "paired_views": "tables/paired_views.csv", "splits": "tables/splits.csv",
              "ruler": "calibration/ruler_photos.csv", "noise_spectrum": "calibration/noise_spectrum.csv",
              "scenarios": "sims/scenarios.csv", "metrics": "sims/metrics.csv", "s1_width_bias": "sims/s1_width_bias.csv",
              "s2_density": "sims/s2_density.csv", "s2_texture": "sims/s2_texture.csv",
              "s4_misclassification": "sims/s4_misclassification.csv", "s6_propagation": "sims/s6_propagation.csv",
              "s7_repeatability": "sims/s7_repeatability.csv", "reproducibility": "qa/reproducibility.csv",
              "platform_stability": "qa/platform_stability.csv"}


# ------------------------------------------------------------------------------------------- helpers
def sha(b):
    return hashlib.sha256(b).hexdigest()


def rnd(x):
    """Round to SIG significant digits (deterministic text, robust to last-digit float noise)."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return float(f"{x:.{SIG}g}")
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return x


def jdump(obj):
    return json.dumps(rnd(obj), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_table(out, name, df):
    cols = [c for c, *_ in S[name]]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    df = df[cols].copy()
    for c, t, *_ in S[name]:
        if t == "float":
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda v: rnd(v) if pd.notna(v) else np.nan)
        elif t == "int":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif t == "bool":
            df[c] = df[c].map(lambda v: None if pd.isna(v) else bool(v))
        else:
            df[c] = df[c].fillna("").astype(str)
    p = out / TABLE_PATH[name]
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator="\n")
    return df


VIEW = {"n": "raw", "e": "enhanced", "g": "grey"}
NAME = re.compile(r"^(?P<who>[a-z0-9]+)-(?P<hand>[LR])?(?P<digit>[a-z]+)-(?P<cap>\d+)-?(?P<suffix>[neg])\.(png|jpg|jpeg|tif|tiff|bmp)$", re.I)
DIGIT = {"pink": "pinky", "index": "pointer"}


def parse_name(fname):
    m = NAME.match(fname)
    if not m:
        return None
    return dict(who=m["who"], hand=m["hand"] or "", digit=DIGIT.get(m["digit"], m["digit"]), capture=int(m["cap"]), suffix=m["suffix"].lower())


# ------------------------------------------------------------------------------------------- load
def load_batch(batch):
    batch = Path(batch)
    bs = json.loads((batch / "batch_summary.json").read_text())
    runs = []
    for i in bs["inputs"]:
        d = batch / i["tag"]
        r = dict(i, dir=d)
        if i["kind"] == "capillary" and (d / "session" / "frames.csv").exists():
            r["frames"] = pd.read_csv(d / "session" / "frames.csv")
            r["loops"] = pd.read_csv(d / "session" / "loops.csv")
            r["per_finger"] = pd.read_csv(d / "per_finger.csv") if (d / "per_finger.csv").exists() else pd.DataFrame()
            r["state"] = json.loads((d / "session_state.json").read_text()) if (d / "session_state.json").exists() else {}
        runs.append(r)
    return bs, runs


def group_sessions(runs):
    """Distinct capillary inputs (identical copies dropped) grouped into sessions: inputs that share capture
    stems (same subject-site-capture, different view) are one session."""
    cap = [r for r in runs if r["kind"] == "capillary" and "frames" in r]
    distinct = [r for r in cap if not r.get("duplicate_of")]
    dups = {}
    for r in cap:
        if r.get("duplicate_of"):
            dups.setdefault(r["duplicate_of"], []).append(r["input"])
    stems = {}
    for r in distinct:
        stems[r["input"]] = {re.sub(r"-?[neg]\.\w+$", "", f, flags=re.I) for f in r["frames"].file}
    groups = []
    for r in sorted(distinct, key=lambda r: r["input"]):
        hit = next((g for g in groups if any(stems[r["input"]] & stems[x["input"]] for x in g)), None)
        (hit.append(r) if hit is not None else groups.append([r]))
    groups.sort(key=lambda g: min(min(x["frames"].file) for x in g))
    return groups, dups


def subject_map(groups):
    whos = []
    for g in groups:
        for r in g:
            for f in sorted(r["frames"].file):
                p = parse_name(f)
                w = p["who"] if p else "unknown"
                if w not in whos:
                    whos.append(w)
    return {w: f"SUBJ-{i + 1:02d}" for i, w in enumerate(whos)}


def site_of(subj, hand, digit):
    return f"{subj}-{hand}{digit}"


def assign_splits(sites):
    """Whole sites go to one split (no frame of a finger leaks across splits). Called once per session, so every
    session feeds val and test. Sites are dealt in the order of sha256(site_id): the first ~15% go to test, the
    next ~15% to val, the rest to train; with >= 3 sites every split gets at least one."""
    order = sorted(sites, key=lambda s: sha(s.encode()))
    n = len(order)
    n_test = max(1, round(n * SPLIT_FRACTIONS[0][1])) if n >= 3 else 0
    n_val = max(1, round(n * SPLIT_FRACTIONS[1][1])) if n >= 3 else 0
    out = {}
    for i, s in enumerate(order):
        out[s] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return out, {s: sha(s.encode())[:12] for s in order}


# ------------------------------------------------------------------------------------------- build
def build(batch, out, compare_with=None):
    out = Path(out)
    bs, runs = load_batch(batch)
    SCALE.clear()
    SCALE.update(scale_of(bs))
    U_ = SCALE["um"]
    groups, dups = group_sessions(runs)
    smap = subject_map(groups)
    alias = {i["input"]: f"IN-{k + 1:02d}" for k, i in enumerate(sorted(bs["inputs"], key=lambda i: (i["sha256"], i["input"])))}
    private = dict(subjects=smap, files={}, inputs={}, input_aliases=alias)

    frames_rows, loops_rows, site_frames, sessions_rows, pv_rows = [], [], {}, [], []
    finger_to_site = {}
    for si, g in enumerate(groups):
        sid = f"S{si + 1:02d}"
        F_all = []
        for r in g:
            private["inputs"][r["input"]] = sid
            F = r["frames"].copy()
            F["_input"] = r["input"]
            L = r["loops"].copy()
            L["_input"] = r["input"]
            F_all.append((r, F, L))
        for r, F, L in F_all:
            from analyze_session import usable as _usable   # the same rule the pipeline used
            U = set(_usable(F).file)
            for _, x in F.sort_values("file").iterrows():
                p = parse_name(x.file) or dict(who="unknown", hand="", digit="unknown", capture=0, suffix="n")
                subj = smap[p["who"]]
                site = site_of(subj, p["hand"], p["digit"])
                finger_to_site[(r["input"], x.finger)] = site
                view = VIEW[p["suffix"]]
                fid = f"{sid}/{subj}-{p['hand']}{p['digit']}-c{p['capture']:02d}-{view}"
                private["files"][fid] = x.file
                reasons = []
                if not x.primary:
                    reasons.append("not_measured_view")
                else:
                    if x.n_resolved < 3:
                        reasons.append("too_few_resolved_loops")
                    if x.glare_frac >= 0.30:
                        reasons.append("glare")
                row = dict(frame_id=fid, session_id=sid, subject_id=subj, site_id=site, hand=p["hand"], digit=p["digit"],
                           capture=p["capture"], view=view, view_by_pixels=x.kind, view_label_matches_pixels=(view == x.kind),
                           measured=bool(x.primary), usable=x.file in U, unusable_reason=";".join(reasons) if x.file not in U else "",
                           n_apices=x.n_apices, n_loops_measured=x.n_loops, n_loops_resolved=x.n_resolved,
                           glare_pct=100 * x.glare_frac, clipped_pct=100 * x.clipped_frac, sharpness=x.sharpness, noise_dn=x.noise_dn,
                           mean_r=x.mean_r, mean_g=x.mean_g, mean_b=x.mean_b, illum_strength=x.illum_strength,
                           blur_sigma_px=x.sigma_px, lumen_width_fit_px=x.lumen_width_px_fit, lumen_contrast=x.lumen_contrast_fit,
                           median_gap_px=x.median_gap_px, median_limb_left_px=x.median_limb_left_px,
                           median_limb_right_px=x.median_limb_right_px, median_apical_px=x.median_apical_px,
                           median_total_px=x.median_total_px)
                row["median_apical_um"] = row["median_apical_px"] * U_
                row["median_total_um"] = row["median_total_px"] * U_
                frames_rows.append(row)
                site_frames.setdefault((sid, site), 0)
                site_frames[(sid, site)] += 1
                lx = L[L.file == x.file].sort_values("apex_x")
                for k, (_, l) in enumerate(lx.iterrows()):
                    lr = dict(loop_id=f"{fid}/L{k + 1:02d}", frame_id=fid, session_id=sid, site_id=site, view=view,
                              frame_usable=x.file in U, apex_x_px=l.apex_x, apex_y_px=l.apex_y,
                              resolved=bool(np.isfinite(l.limb_left) and np.isfinite(l.limb_right)),
                              limb_left_px=l.limb_left, limb_right_px=l.limb_right, apical_px=l.apical, total_px=l.total,
                              depth_frac=l.depth_frac)
                    for c in ("limb_left", "limb_right", "apical", "total"):
                        lr[f"{c}_um"] = lr[f"{c}_px"] * U_
                    loops_rows.append(lr)
        Fs = [f for f in frames_rows if f["session_id"] == sid]
        sessions_rows.append(dict(session_id=sid, subjects=";".join(sorted({f["subject_id"] for f in Fs})),
                                  views=";".join(sorted({f["view"] for f in Fs})), n_inputs=len(g),
                                  n_duplicate_inputs=sum(len(dups.get(r["input"], [])) for r in g),
                                  input_sha256=";".join(r["sha256"] for r in g), n_frames=len(Fs),
                                  n_measured_frames=sum(f["measured"] for f in Fs), n_usable_frames=sum(f["usable"] for f in Fs),
                                  n_loops_resolved=sum(1 for l in loops_rows if l["session_id"] == sid and l["frame_usable"] and l["resolved"]),
                                  um_per_px=U_, um_per_px_low=SCALE["low"], um_per_px_high=SCALE["high"],
                                  scale_status=SCALE["status"], scale_source=SCALE["source"]))

    split_of, rank = {}, {}
    for sid in sorted({sid for sid, _ in site_frames}):          # stratified: every session feeds val and test
        a, b = assign_splits(sorted({s for x, s in site_frames if x == sid}))
        split_of.update(a)
        rank.update(b)
    for rows in (frames_rows, loops_rows):
        for r in rows:
            r["split"] = split_of[r["site_id"]]
    splits = pd.DataFrame([dict(site_id=s, session_id=sid, split=split_of[s], n_frames=n, rank_key=rank[s])
                           for (sid, s), n in sorted(site_frames.items())])

    # sites: per input's per_finger table, view = majority view of that input's measured frames
    site_rows = []
    for si, g in enumerate(groups):
        sid = f"S{si + 1:02d}"
        for r in g:
            vs = pd.Series([VIEW[(parse_name(f) or {"suffix": "n"})["suffix"]] for f in r["frames"][r["frames"].primary].file])
            view = vs.mode().iloc[0] if len(vs) else "raw"
            for _, x in r["per_finger"].iterrows():
                site = finger_to_site.get((r["input"], x.finger))
                if site is None:
                    continue
                site_rows.append(dict(site_id=site, session_id=sid, view=view, split=split_of[site], measured_frames=x.primary_frames,
                                      usable_frames=x.usable_frames, usable_pct=x.usable_pct, loops_detected=x.loops_detected,
                                      loops_resolved=x.loops_resolved, limb_px_median=x.limb_px_median, apical_px_median=x.apical_px_median,
                                      total_px_median=x.total_px_median, blur_sigma_px=x.sigma_px, lumen_contrast=x.contrast,
                                      illum_strength=x.illum, glare_pct=x.glare_pct, sharpness=x.sharpness, gap_px=x.gap_px,
                                      apical_cv_pct_between_captures=x.apical_cv_pct_between_captures))

    # paired views
    pp = Path(batch) / "paired_raw_vs_enhanced.csv"
    if pp.exists():
        P = pd.read_csv(pp)
        canon = {r["input"] for g in groups for r in g}
        P = P[P.zip_raw.isin(canon) & P.zip_enh.isin(canon)]
        for _, x in P.sort_values("capture").iterrows():
            pr = parse_name(x.raw_file) or {}
            subj = smap.get(pr.get("who"), "SUBJ-??")
            site = site_of(subj, pr.get("hand", ""), pr.get("digit", ""))
            sid = private["inputs"][x.zip_raw]
            row = dict(capture_id=f"{sid}/{subj}-{pr.get('hand', '')}{pr.get('digit', '')}-c{pr.get('capture', 0):02d}",
                       session_id=sid, site_id=site, split=split_of.get(site, ""), paired_by=x.paired_by)
            for v, pre in (("raw", "raw_"), ("enhanced", "enh_")):
                row.update({f"{v}_n_apices": x[pre + "n_apices"], f"{v}_n_loops_resolved": x[pre + "n_resolved"],
                            f"{v}_sharpness": x[pre + "sharpness"], f"{v}_noise_dn": x[pre + "noise_dn"],
                            f"{v}_blur_sigma_px": x[pre + "sigma_px"], f"{v}_median_apical_px": x[pre + "median_apical_px"],
                            f"{v}_glare_pct": 100 * x[pre + "glare_frac"]})
            pv_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    T = {}
    T["sessions"] = write_table(out, "sessions", pd.DataFrame(sessions_rows))
    T["frames"] = write_table(out, "frames", pd.DataFrame(frames_rows).sort_values("frame_id"))
    T["loops"] = write_table(out, "loops", pd.DataFrame(loops_rows).sort_values("loop_id"))
    T["sites"] = write_table(out, "sites", pd.DataFrame(site_rows).sort_values(["site_id", "view"]))
    T["paired_views"] = write_table(out, "paired_views", pd.DataFrame(pv_rows, columns=[c for c, *_ in S["paired_views"]]))
    T["splits"] = write_table(out, "splits", splits.sort_values("site_id"))

    # calibration
    cal = bs["calibration"]
    nail_sha = {i["sha256"] for i in bs["inputs"] if i["kind"] in ("ruler", "calibration")
                and ((i.get("result") or {}).get("ruler", i.get("result")) or {}).get("nail_zoom")}
    Rs = []
    for r in sorted([r for r in runs if r["kind"] in ("ruler", "calibration") and not r.get("duplicate_of")], key=lambda r: alias[r["input"]]):
        if not (r["dir"] / "ruler_frames.csv").exists():
            continue
        R = pd.read_csv(r["dir"] / "ruler_frames.csv")
        R["input_id"] = alias[r["input"]]
        R["photo"] = [str(i + 1) for i in range(len(R))]
        R["nail_zoom"] = r["sha256"] in nail_sha
        R = R.rename(columns={"se": "se_um_per_px", "um_per_px_peakpicking_old": "um_per_px_old_peakpicking", "tick_sigma_px": "tick_blur_sigma_px"})
        if "method" not in R:
            R["method"] = "tick-band autocorrelation"
        Rs.append(R)
    if Rs:
        R = pd.concat(Rs, ignore_index=True)
        for c, *_ in S["ruler"]:
            if c not in R:
                R[c] = np.nan
        T["ruler"] = write_table(out, "ruler", R)
    used = ((cal.get("noise_used") or {}).get("source") or "")
    braw = [r for r in runs if r["kind"] in ("blank", "calibration") and (r["dir"] / "noise_spectrum.csv").exists() and not r.get("duplicate_of")]
    bpick = next((r for r in braw if f"blank card {r['input']}," in used), braw[0] if braw else None)
    if bpick:
        T["noise_spectrum"] = write_table(out, "noise_spectrum", pd.read_csv(bpick["dir"] / "noise_spectrum.csv"))
    if SCALE["status"] == "measured":
        note = (f"Every um value uses {SCALE['um']:.4f} um/px, measured from the calibration card photographed at the nail zoom "
                f"({SCALE['n']} photos; both arms of the cross; vertical/horizontal pitch ratio {SCALE['aspect']:.4f}). A card photographed "
                "at a lower zoom (nail_zoom false in ruler_photos.csv) is kept only as a record and for the ruler_zoom what-if scenario. "
                "The card's own printing accuracy has not been checked against a certified micrometer.")
    else:
        note = ("No card was photographed at the nail zoom: every um column uses the anatomical estimate in sessions.csv. "
                "Photograph the card at the exact nail zoom and height to replace it.")
    calib = dict(scale=dict(um_per_px=SCALE["um"], low=SCALE["low"], high=SCALE["high"], status=SCALE["status"], n_photos=SCALE["n"],
                            card_blur_sigma_px=SCALE.get("tick_sigma_px"), aspect_v_over_h=SCALE.get("aspect")),
                 low_zoom_ruler=cal.get("ruler"), blank_card={k: v for k, v in (cal.get("blank") or {}).items() if k != "corr_fit"},
                 noise_model_used=dict(cal.get("noise_used") or {}, source="blank card, one exposure level"),
                 cards=[dict({k: v for k, v in c.items() if k != "input"}, input_id=alias.get(c["input"], "")) for c in cal.get("rulers_all", [])],
                 note=note)
    (out / "calibration").mkdir(parents=True, exist_ok=True)
    (out / "calibration" / "calibration.json").write_text(jdump(calib))

    # sims
    scen_rows, met_rows, curves = [], [], {k: [] for k in ("s1_width_bias", "s2_density", "s2_texture", "s4_misclassification", "s6_propagation", "s7_repeatability")}
    stab = compare_platforms(batch, compare_with, groups) if compare_with else {}
    mode = bs.get("sims", "quick")
    for si, g in enumerate(groups):
        sid = f"S{si + 1:02d}"
        for r in g:
            vs = pd.Series([VIEW[(parse_name(f) or {"suffix": "n"})["suffix"]] for f in r["frames"][r["frames"].primary].file])
            view = vs.mode().iloc[0] if len(vs) else "raw"
            for scen, path in sorted((r.get("state") or {}).get("scopes", {}).items()):
                sp = json.loads((r["dir"] / "scopes" / f"{scen}.json").read_text())
                site = ""
                name = scen
                if scen.startswith("finger_"):
                    site = finger_to_site.get((r["input"], scen[len("finger_"):]), "")
                    name = f"site_{site}"
                scid = f"{sid}-{view}-{name}"
                scen_rows.append(dict(scenario_id=scid, session_id=sid, view=view, scenario=name, site_id=site, um_per_px=sp["um_per_px"],
                                      sigma_um=sp["sigma_um"], contrast=sp["contrast"], illum_strength=sp["illum_strength"],
                                      photons=sp["photons"], corr_noise=json.dumps(rnd(sp.get("corr_noise"))) if sp.get("corr_noise") else "",
                                      texture_amp=sp.get("texture_amp"),
                                      scale_status=("ruler" if scen == "ruler_zoom" else
                                                    "measured" if (sp.get("measured") or {}).get("um_per_px") else "anatomical_estimate")))
                m = json.loads((r["dir"] / "sim_metrics.json").read_text()).get(scen, {}) if (r["dir"] / "sim_metrics.json").exists() else {}
                met_rows.append(dict(scenario_id=scid, mode=mode,
                                     **{k: m.get(k) for k in ("s1_limb_mean_abs_bias_pct", "s1_apical_mean_abs_bias_pct", "s1_total_mean_abs_bias_pct",
                                                              "s1_limb_worst_bias_pct", "s1_apical_worst_bias_pct", "s1_total_worst_bias_pct",
                                                              "s1_min_limb_under_10pct_um", "s4_max_sigma_um_for_5pct", "s4_p7_called_reduced",
                                                              "s4_p5_called_normal", "s4_density_est_9", "s7_icc", "s7_cv_pct")},
                                     s4_giant_error=m.get("s4_giant_error_at_scope_sigma"),
                                     platform_unstable=";".join(sorted(stab.get(("unstable", r["input"], scen), [])))))
                sd = r["dir"] / "sims" / scen
                for f, key, ren in (("s1/s1_width_bias.csv", "s1_width_bias", {"true": "true_um", "sigma": "sigma_um", "pixel": "pixel_um", "mean": "mean_um", "sd": "sd_um"}),
                                    ("s2/s2_density.csv", "s2_density", {"sigma": "sigma_um"}),
                                    ("s2/s2_texture.csv", "s2_texture", {}),
                                    ("s4/s4_misclassification.csv", "s4_misclassification", {"sigma": "sigma_um"}),
                                    ("s6/s6_propagation.csv", "s6_propagation", {"true_width": "true_width_um", "mean": "mean_pct", "sd": "sd_pct", "p5": "p5_pct", "median": "median_pct", "p95": "p95_pct"})):
                    if (sd / f).exists():
                        d = pd.read_csv(sd / f).rename(columns=ren)
                        d.insert(0, "scenario_id", scid)
                        curves[key].append(d)
                if (sd / "s7" / "summary.json").exists():
                    s7 = json.loads((sd / "s7" / "summary.json").read_text()).get("s7", {})
                    ci = s7.get("ci") or [None, None]
                    curves["s7_repeatability"].append(pd.DataFrame([dict(scenario_id=scid, icc=s7.get("icc"), ci_lo=ci[0], ci_hi=ci[1],
                                                                         icc_oracle_position=s7.get("icc_oracle_position"), cv_pct=s7.get("cv_pct"),
                                                                         n_subjects=s7.get("n"))]))
    T["scenarios"] = write_table(out, "scenarios", pd.DataFrame(scen_rows, columns=[c for c, *_ in S["scenarios"]]).sort_values("scenario_id"))
    T["metrics"] = write_table(out, "metrics", pd.DataFrame(met_rows, columns=[c for c, *_ in S["metrics"]]).sort_values("scenario_id"))
    for k, parts in curves.items():
        d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[c for c, *_ in S[k]])
        sort = [c for c in [c for c, *_ in S[k]] if c in d.columns]
        T[k] = write_table(out, k, d.sort_values(sort, kind="mergesort") if len(d) else d)

    # qa
    T["reproducibility"] = write_table(out, "reproducibility", pd.DataFrame(
        [dict(input_a=alias[d["a"]], input_b=alias[d["b"]], output_files=d["files_compared"], byte_identical=d.get("byte_identical"),
              float_equal=d.get("float_equal"), real_differences="; ".join(f"{k}: {v}" for k, v in d.get("real_differences", {}).items()))
         for d in bs.get("determinism", [])], columns=[c for c, *_ in S["reproducibility"]]))
    if stab:
        T["platform_stability"] = write_table(out, "platform_stability", stab["table"])

    # llm + vision labels
    write_llm(out, T)
    (out / "_private").mkdir(exist_ok=True)
    (out / "_private" / "README.txt").write_text("Real subject names and original file names. Never commit or share this folder.\n")
    (out / "_private" / "id_map.json").write_text(json.dumps(private, indent=2, sort_keys=True) + "\n")

    write_card(out, bs, T, groups, dups, stab)
    write_schema_md(out)
    write_manifest(out, bs)
    problems = check(out)
    if problems:
        raise SystemExit("dataset check failed:\n  " + "\n  ".join(problems))
    print(f"-> {out}  ({len(T['frames'])} frames, {len(T['loops'])} loops, {len(T['scenarios'])} scenarios)")
    return T


def compare_platforms(batch, other, groups):
    """Headline sim metrics from the same inputs run on a second machine: which ones moved."""
    rows, unstable = [], {}
    for g in groups:
        for r in g:
            a, b = Path(batch) / r["tag"] / "sim_metrics.json", Path(other) / r["tag"] / "sim_metrics.json"
            if not (a.exists() and b.exists()):
                continue
            ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
            for scen in ma:
                for k, v in ma[scen].items():
                    w = mb.get(scen, {}).get(k)
                    if isinstance(v, (int, float)) and isinstance(w, (int, float)):
                        d = abs(v - w)
                        rows.append(dict(metric=k, diff=d))
                        if d > 1e-9 * max(1.0, abs(v)):
                            unstable.setdefault(("unstable", r["input"], scen), set()).add(k)
    if not rows:
        return {}
    D = pd.DataFrame(rows)
    tab = D.groupby("metric")["diff"].agg(n_scenarios="count", n_differ=lambda x: int((x > 1e-9).sum()),
                                          max_abs_diff="max", median_abs_diff="median").reset_index().sort_values("metric")
    return dict(unstable, table=tab)


# ------------------------------------------------------------------------------------------- llm
def _f(x, nd=1):
    return "unknown" if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else f"{float(x):.{nd}f}"


def frame_text(f):
    s = (f"{f['frame_id']}: {f['view']} frame, {f['digit']} {('(' + f['hand'] + ' hand) ') if f['hand'] else ''}"
         f"capture {f['capture']}. Glare {_f(f['glare_pct'])}%, sharpness {_f(f['sharpness'])}, noise {_f(f['noise_dn'], 2)} DN, "
         f"blur {_f(f['blur_sigma_px'], 2)} px. {int(f['n_apices'])} apexes detected, {int(f['n_loops_resolved'])} loops with both limbs resolved.")
    if f["usable"]:
        s += (f" Usable. Median apical width {_f(f['median_apical_px'])} px (about {_f(f['median_apical_um'])} um at the "
              f"{'measured' if SCALE.get('status') == 'measured' else 'estimated'} scale), total width {_f(f['median_total_px'])} px.")
    else:
        s += f" Not usable: {f['unusable_reason'].replace('_', ' ').replace(';', ', ')}."
    return s


def qc_answer(f):
    if f["usable"]:
        return (f"Usable. {int(f['n_loops_resolved'])} loops have both limbs resolved (3 needed) and glare is {_f(f['glare_pct'])}% (under 30%). "
                + ("Measure on this raw frame." if f["view"] == "raw" else "It is the processed view, so prefer the raw frame of the same capture when it exists: enhancement roughly doubles pixel noise and reads widths slightly narrower."))
    tips = []
    for reason in f["unusable_reason"].split(";"):
        if reason == "not_measured_view":
            tips.append("this is the second view of a capture whose raw frame is measured instead")
        elif reason == "too_few_resolved_loops":
            n = int(f['n_loops_resolved'])
            tips.append(f"{'no loop has' if n == 0 else ('only 1 loop has' if n == 1 else f'only {n} loops have')} both limbs resolved (3 needed): refocus on the apical limbs, press lighter and keep the scope square to the nailfold")
        elif reason == "glare":
            tips.append(f"glare covers {_f(f['glare_pct'])}% of the frame (limit 30%): move the LED reflection to the edge or add a thin drop of oil")
    return "Not usable: " + "; ".join(tips) + "."


def write_llm(out, T):
    F = T["frames"].to_dict("records")
    L = T["loops"]
    M = T["metrics"].set_index("scenario_id") if len(T["metrics"]) else pd.DataFrame()
    SC = T["scenarios"].set_index("scenario_id") if len(T["scenarios"]) else pd.DataFrame()
    d = out / "llm"
    d.mkdir(parents=True, exist_ok=True)
    recs = []
    for f in sorted(F, key=lambda r: r["frame_id"]):
        lp = L[L.frame_id == f["frame_id"]]
        recs.append(dict(id=f["frame_id"], split=f["split"], session_id=f["session_id"], site_id=f["site_id"], view=f["view"],
                         usable=f["usable"], unusable_reason=f["unusable_reason"],
                         metrics={k: rnd(f[k]) for k in ("n_apices", "n_loops_resolved", "glare_pct", "sharpness", "noise_dn", "blur_sigma_px",
                                                         "lumen_contrast", "median_apical_px", "median_total_px", "median_gap_px") if pd.notna(f[k])},
                         loops=[{k: rnd(v) for k, v in l.items() if k in ("apex_x_px", "apex_y_px", "resolved", "limb_left_px", "limb_right_px", "apical_px", "total_px") and pd.notna(v)}
                                for l in lp.to_dict("records")],
                         text=frame_text(f)))
    (d / "records.jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in recs))

    SP = system_prompt()

    def chat(task, split, user, assistant, source):
        return dict(task=task, split=split, source=source,
                    messages=[dict(role="system", content=SP), dict(role="user", content=user), dict(role="assistant", content=assistant)])

    ex = []
    for f in sorted(F, key=lambda r: r["frame_id"]):
        ex.append(chat("frame_qc", f["split"], (
            f"Is this capture usable for measurement? View: {f['view']}. Glare: {_f(f['glare_pct'])}% of the frame. "
            f"Loop apexes detected: {int(f['n_apices'])}. Loops with both limbs resolved: {int(f['n_loops_resolved'])}. "
            f"Sharpness: {_f(f['sharpness'])}. Noise: {_f(f['noise_dn'], 2)} DN. Measured view for this capture: {'yes' if f['measured'] else 'no'}."),
            qc_answer(f), f["frame_id"]))
    for l in L[(L.frame_usable == True) & (L.resolved == True)].sort_values("loop_id").to_dict("records"):  # noqa: E712
        scen = f"{l['session_id']}-{l['view']}-site_{l['site_id']}"
        fb = next((f"{l['session_id']}-{l['view']}-{k}" for k in ("measured", "anat_central") if f"{l['session_id']}-{l['view']}-{k}" in M.index), None)
        m = M.loc[scen] if scen in M.index else (M.loc[fb] if fb else None)
        sc = SC.loc[scen] if scen in SC.index else None
        bias = "" if m is None else (f" At this site's measured blur the simulator puts the typical bias at about {_f(m['s1_limb_mean_abs_bias_pct'])}% for limbs, "
                                     f"{_f(m['s1_apical_mean_abs_bias_pct'])}% for apical width and {_f(m['s1_total_mean_abs_bias_pct'])}% for total width"
                                     + (" (quick-mode figure that varies between platforms; treat as approximate)." if isinstance(m["platform_unstable"], str) and m["platform_unstable"] else "."))
        px_txt = (f"Limb widths {_f(l['limb_left_px'], 2)} px and {_f(l['limb_right_px'], 2)} px, "
                  f"apical width {_f(l['apical_px'], 2)} px, total width {_f(l['total_px'], 2)} px.")
        if SCALE["status"] == "measured":
            ex.append(chat("loop_measurement", l["split"], (
                f"Report this capillary loop in micrometres. {px_txt} The calibration card photographed at this zoom gives "
                f"{SCALE['um']:.4f} um per pixel."),
                (f"At {SCALE['um']:.4f} um/px: limbs {_f(l['limb_left_um'])} and {_f(l['limb_right_um'])} um, apical {_f(l['apical_um'])} um, "
                 f"total {_f(l['total_um'])} um. The scale itself is tight (the card photos agree to about 0.01%); the uncertainty is in "
                 f"the width measurement." + bias), l["loop_id"]))
        else:
            ex.append(chat("loop_measurement", l["split"], (
                f"Report this capillary loop in micrometres. {px_txt} The scale has not been calibrated at this zoom."),
                (f"Using the anatomical scale estimate of {UM_PER_PX_EST} um/px (plausible range {UM_LOW} to {UM_HIGH}): limbs about "
                 f"{_f(l['limb_left_um'])} and {_f(l['limb_right_um'])} um, apical about {_f(l['apical_um'])} um "
                 f"(range {_f(l['apical_px'] * UM_LOW)} to {_f(l['apical_px'] * UM_HIGH)}), total about {_f(l['total_um'])} um. "
                 f"These are estimates until the card is photographed at the nail zoom." + bias), l["loop_id"]))
    for scid, sc in (SC.iterrows() if len(SC) else []):
        if scid not in M.index:
            continue
        m = M.loc[scid]
        caveat = (f" Quick-mode values; {m['platform_unstable'].replace(';', ', ')} changed between two platforms, so treat those as approximate."
                  if isinstance(m["platform_unstable"], str) and m["platform_unstable"] else "")
        ex.append(chat("simulation_lookup", "train", (
            f"What measurement errors should I expect with {_f(sc['um_per_px'], 2)} um per pixel, blur sigma {_f(sc['sigma_um'], 2)} um, "
            f"lumen contrast {_f(sc['contrast'], 2)} and {'the measured blotchy camera noise' if sc['corr_noise'] else 'white noise only'}?"),
            (f"In simulation: limb width bias about {_f(m['s1_limb_mean_abs_bias_pct'])}%, apical {_f(m['s1_apical_mean_abs_bias_pct'])}%, "
             f"total {_f(m['s1_total_mean_abs_bias_pct'])}% (mean absolute over the S1 grid). Telling a 45 um loop from a 55 um giant loop fails "
             f"{_f(100 * m['s4_giant_error'] if pd.notna(m['s4_giant_error']) else None)}% of the time. Repeat captures vary by about {_f(m['s7_cv_pct'])}% "
             f"(simulated ICC {_f(m['s7_icc'], 2)}). Widths smaller than {_f(m['s1_min_limb_under_10pct_um'], 0)} um are biased more than 10%." + caveat), scid))
    PV = T["paired_views"]
    if len(PV):
        nz = PV[["raw_noise_dn", "enhanced_noise_dn"]].dropna()
        ap = PV[["raw_median_apical_px", "enhanced_median_apical_px"]].dropna()
        lr = PV[["raw_n_loops_resolved", "enhanced_n_loops_resolved"]].dropna()
        noise_x = float(nz.enhanced_noise_dn.mean() / nz.raw_noise_dn.mean()) if len(nz) else float("nan")
        ap_pct = float(100 * (ap.enhanced_median_apical_px.mean() / ap.raw_median_apical_px.mean() - 1)) if len(ap) else float("nan")
        lr_d = float((lr.enhanced_n_loops_resolved - lr.raw_n_loops_resolved).mean()) if len(lr) else float("nan")
        why = (f"Measure the raw view. Over the {len(PV)} captures in this dataset seen in both views, the enhanced view has "
               f"{noise_x:.1f} times the pixel noise, resolves {abs(lr_d):.1f} {'fewer' if lr_d < 0 else 'more'} loops per frame on average and reads "
               f"apical width {abs(ap_pct):.0f}% {'narrower' if ap_pct < 0 else 'wider'}, so it adds bias without adding information.")
    for p in T["paired_views"].to_dict("records"):
        ex.append(chat("raw_vs_enhanced", p["split"], (
            f"Same capture in two views. Raw: noise {_f(p['raw_noise_dn'], 2)} DN, {int(p['raw_n_loops_resolved'])} loops resolved, apical "
            f"{_f(p['raw_median_apical_px'])} px. Enhanced: noise {_f(p['enhanced_noise_dn'], 2)} DN, {int(p['enhanced_n_loops_resolved'])} loops resolved, "
            f"apical {_f(p['enhanced_median_apical_px'])} px. Which should I measure?"),
            why, p["capture_id"]))
    if "ruler" in T and len(T["ruler"]):
        Fu = T["frames"][T["frames"].usable == True]   # noqa: E712
        tot_px = float(Fu.median_total_px.median()) if len(Fu) else float("nan")
        gap_px = float(Fu.median_gap_px.median()) if len(Fu) else float("nan")
        nail = [r for r in T["ruler"].to_dict("records") if r["nail_zoom"] is True and pd.notna(r["um_per_px"])]
        for r in T["ruler"].sort_values(["input_id", "photo"]).to_dict("records"):
            if pd.isna(r["um_per_px"]):
                continue
            asp = r.get("aspect_v_over_h")
            q = (f"A photo of the Captrail calibration card (0.1 mm between small ticks, 1 mm between long lines) shows the small ticks "
                 f"{_f(r['pitch_px'], 2)} px apart" + (f", and the vertical arm's ticks {_f(asp, 4)} times the horizontal spacing" if pd.notna(asp) else "")
                 + ". What is the scale, and can I use it for my nail images?")
            u = float(r["um_per_px"])
            if r["nail_zoom"] is True:
                a = (f"100 um / {_f(r['pitch_px'], 2)} px = {u:.4f} um per pixel" + (", and the two arms agree, so the pixels are square" if pd.notna(asp) and abs(asp - 1) < 0.005 else "")
                     + f". Yes, provided the photo was taken at exactly the zoom, height and focus of the nail captures. A 640 x 480 frame then spans "
                       f"{640 * u / 1000:.2f} x {480 * u / 1000:.2f} mm, and this dataset's median loop is {tot_px * u:.0f} um across with loops "
                       f"{gap_px * u:.0f} um apart, which fits healthy anatomy.")
            else:
                ref = (f" Use the card photographed at nail zoom instead ({nail[0]['um_per_px']:.4f} um/px)." if nail else
                       " Photograph the card again at the exact nail zoom and height.")
                a = (f"100 um / {_f(r['pitch_px'], 2)} px = {u:.3f} um per pixel, but not for the nail images: this photo was taken at a lower zoom. "
                     f"At {u:.2f} um/px this dataset's median loop would be {tot_px * u:.0f} um across with loops {gap_px * u:.0f} um apart, far "
                     "outside healthy anatomy, so the nail captures were taken at a higher zoom." + ref)
            ex.append(chat("calibration_qc", "train", q, a, f"{r['input_id']}/photo{r['photo']}"))
    for s in ("train", "val", "test"):
        rows = [e for e in ex if e["split"] == s]
        (d / f"sft_{s}.jsonl").write_text("".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in rows))


# ------------------------------------------------------------------------------------------- docs
def write_schema_md(out):
    L = ["# Schema", "", "Every CSV column: type, unit and meaning. `*_um` columns use the scale in `tables/sessions.csv` "
         "(`scale_status` says whether it was measured at nail zoom or estimated).", ""]
    for name, cols in S.items():
        if not (out / TABLE_PATH[name]).exists():
            continue
        L += [f"## `{TABLE_PATH[name]}`", "", "| column | type | unit | meaning |", "|---|---|---|---|"]
        L += [f"| `{c}` | {t} | {u} | {d} |" for c, t, u, d in cols]
        L.append("")
    L += ["## `llm/records.jsonl`", "", "One JSON object per frame: `id`, `split`, `session_id`, `site_id`, `view`, `usable`, `unusable_reason`,",
          "`metrics` (frame numbers), `loops` (per-loop numbers), `text` (the same facts as one paragraph).", "",
          "## `llm/sft_{train,val,test}.jsonl`", "", "Chat-format supervised examples: `messages` (system, user, assistant), `task`, `split`, `source` (the frame,",
          "loop, scenario, capture or card photo the answer was computed from). Tasks: `frame_qc`, `loop_measurement`, `simulation_lookup`,",
          "`raw_vs_enhanced`, `calibration_qc`.",
          "Answers are generated by fixed templates from the pipeline's own numbers and rules.", ""]
    (out / "SCHEMA.md").write_text("\n".join(L))


def density_note(F):
    """Loop density from the median apex spacing of usable measured frames, checked against the healthy >= 7/mm."""
    Fu = F[(F.usable == True) & (F.measured == True)]   # noqa: E712
    gap = float(Fu.median_gap_px.median()) * SCALE["um"] if len(Fu) else float("nan")
    if not np.isfinite(gap) or gap <= 0:
        return "- **Loop density not available.**"
    per_mm = 1000.0 / gap
    if per_mm < 7:
        return (f"- **Loop density is not reliable.** Detected apexes are a median {gap:.0f} um apart (about {per_mm:.1f} per mm) in "
                "healthy volunteers, below the 7 per mm normal cut-off: the detector misses faint or out-of-focus loops. Do not train "
                "on, or quote, density from this data.")
    return f"- **Loop density.** Detected apexes are a median {gap:.0f} um apart (about {per_mm:.1f} per mm)."


def write_card(out, bs, T, groups, dups, stab):
    F, Lp, Sp = T["frames"], T["loops"], T["splits"]
    ruler_um = ((bs.get("calibration") or {}).get("ruler") or {}).get("um_per_px", {}).get("mean")
    ss = T["sessions"]
    llm = {s: sum(1 for _ in open(out / "llm" / f"sft_{s}.jsonl")) for s in ("train", "val", "test")}
    tasks = pd.Series([json.loads(l)["task"] for s in ("train", "val", "test") for l in open(out / "llm" / f"sft_{s}.jsonl")]).value_counts()
    L = [f"# Captrail capillaroscopy simulation dataset v{VERSION}", "",
         "Measurements and simulation results from a low-cost USB nailfold capillaroscope (Jiusion 40-1000x USB digital microscope,",
         "640 x 480, on the Captrail cradle), produced by the",
         "Captrail `sims` pipeline. Numbers only: no images, subjects pseudonymized. **Research prototype, not diagnostic.**", "",
         "## What is in it", "",
         "| file | rows | one row per |", "|---|---|---|"]
    for name, p in TABLE_PATH.items():
        if (out / p).exists():
            n = len(pd.read_csv(out / p))
            L.append(f"| `{p}` | {n} | {ROW_DESC[name]} |")
    L += [f"| `llm/records.jsonl` | {len(F)} | frame, as structured fields plus text |",
          f"| `llm/sft_train.jsonl` / `val` / `test` | {llm['train']} / {llm['val']} / {llm['test']} | chat example |",
          "| `calibration/calibration.json` | | ruler and blank-card summary |", "",
          "Columns, types and units: `SCHEMA.md`. Hashes of every file and the exact source run: `manifest.json`.", "",
          "## Sessions", "", "| session | subjects | views | frames | usable measured frames | resolved loops | duplicate inputs ignored |", "|---|---|---|---|---|---|---|"]
    for _, s in ss.iterrows():
        L.append(f"| {s.session_id} | {s.subjects} | {s.views} | {s.n_frames} | {s.n_usable_frames}/{s.n_measured_frames} | {s.n_loops_resolved} | {s.n_duplicate_inputs} |")
    L += ["", "## Splits", "",
          "Whole sites (subject + hand + digit) go to one split, so no finger appears in two splits. Within each session, sites are",
          "dealt in the order of sha256(site_id): about 15% test, 15% val, the rest train, at least one site each, so every session",
          "appears in every split. Scenario lookups (`simulation_lookup`) are all train.", "",
          "| split | sites | frames | loops |", "|---|---|---|---|"]
    for s in ("train", "val", "test"):
        L.append(f"| {s} | {int((Sp.split == s).sum())} | {int((F.split == s).sum())} | {int((Lp.split == s).sum())} |")
    L += ["", "## LLM files", "", "| task | examples | what the model learns |", "|---|---|---|"]
    desc = dict(frame_qc="decide if a capture is usable and say how to fix it", loop_measurement="turn pixel widths into micrometres at the right scale, with the right caveats",
                simulation_lookup="expected error for a given scale, blur, contrast and noise", raw_vs_enhanced="which view to measure and why",
                calibration_qc="read a card photo's scale and say whether it applies to the nail images")
    for k in ("frame_qc", "loop_measurement", "simulation_lookup", "raw_vs_enhanced", "calibration_qc"):
        L.append(f"| `{k}` | {int(tasks.get(k, 0))} | {desc[k]} |")
    L += ["", "Fine-tune on `sft_train.jsonl`, pick checkpoints on `sft_val.jsonl`, report once on `sft_test.jsonl`. `records.jsonl` is for",
          "retrieval (embed `text`, filter on `split`). Every answer comes from the pipeline's numbers and rules, so a model trained here learns",
          "Captrail's pipeline, not clinical judgement.", "",
          "## Limits (read before training or quoting)", "",
          "- **Not validated.** No human rater has measured these loops yet (S5). Labels are the pipeline's output.",
          (f"- **Scale is measured at nail zoom.** Every `*_um` column uses {SCALE['um']:.4f} um/px from the calibration card photographed at the "
           f"nail zoom ({SCALE['n']} photos, both arms of the cross, vertical/horizontal {SCALE['aspect']:.4f}; card blur {_f(SCALE.get('tick_sigma_px'), 2)} px). "
           "It holds only for captures at that exact zoom, height and focus. The card's printing accuracy has not been checked against a "
           "certified micrometer." + (f" An earlier card photo at a lower zoom ({ruler_um:.3f} um/px) is kept in `calibration/ruler_photos.csv` "
                                      "with nail_zoom = false." if ruler_um else "")
           if SCALE.get("status") == "measured" else
           f"- **Scale is estimated.** `*_um` uses {UM_PER_PX_EST} um/px (bounds {UM_LOW} to {UM_HIGH}) from healthy anatomy (loop spacing and widths). "
           "Photograph the card at the exact nail zoom to replace the estimate."),
          f"- **Small and healthy.** {len(set(F.subject_id))} subjects, all healthy volunteers; no disease labels. Do not train a diagnostic classifier on this.",
          density_note(F),
          ("- **Quick-mode sims.** Few replicates per scenario. `sims/metrics.csv` column `platform_unstable` lists metrics that changed between two "
           "platforms; see `qa/platform_stability.csv`." if stab else
           "- **Quick-mode sims.** Few replicates per scenario, all computed on one platform. A cross-platform check on dataset v1 found S1 apical "
           "bias and S7 repeatability can move between machines (up to 6 points and 0.29 ICC); S4 decisions and density calls did not. Treat S1 "
           "and S7 per-scenario values as approximate."),
          "- **Auto labels.** Apex positions and widths in `tables/loops.csv` are detector output, useful as pseudo-labels for a detector after a human check.", "",
          "## Reproduce", "", "```", "python batch_run.py <folder of capture zips> --out results/batch",
          f"python export_dataset.py results/batch --out datasets/{out.name}", f"python export_dataset.py --check datasets/{out.name}", "```", "",
          f"Source run: batch code `{bs['code_sha256'][:12]}`, python {bs['versions']['python']}, numpy {bs['versions']['numpy']}, "
          f"scipy {bs['versions']['scipy']}, {bs['versions']['platform'].split('-')[0]}. The same inputs give byte-identical outputs on one machine "
          "(`qa/reproducibility.csv`)."
          + (f" {bs.get('units_from_other_machine')} sim steps were computed on a second machine (Linux) from byte-identical scope files and "
             "merged by cache key, so all sims in this release come from one platform." if bs.get('units_from_other_machine') else ""), "",
          "Images stay on the team's computers. To build an image + keypoint set locally (never committed):", "",
          "```", f"python export_dataset.py results/batch --out datasets/{out.name} --vision-images <folder of zips> --vision-out datasets/_local_vision", "```", ""]
    (out / "README.md").write_text("\n".join(L))


def write_manifest(out, bs):
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and "_private" not in p.parts and p.name != "manifest.json":
            rel = str(p.relative_to(out))
            info = dict(sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
            if p.suffix == ".csv":
                info["rows"] = int(len(pd.read_csv(p)))
            elif p.suffix == ".jsonl":
                info["rows"] = sum(1 for _ in open(p))
            files[rel] = info
    man = dict(dataset="captrail-capillaroscopy-sims", version=VERSION,
               exporter_sha256=sha((ROOT / "export_dataset.py").read_bytes()),
               source=dict(batch_code_sha256=bs["code_sha256"], sims_mode=bs.get("sims"), versions=bs["versions"],
                           inputs=[dict(sha256=i["sha256"], kind=i["kind"], duplicate=bool(i.get("duplicate_of"))) for i in bs["inputs"]]),
               files=files)
    (out / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------- check
def schema_from_md(out):
    """{csv path: [columns]} from a dataset's own SCHEMA.md (used to check releases made by an older exporter)."""
    exp, cur = {}, None
    for line in (Path(out) / "SCHEMA.md").read_text().splitlines():
        m = re.match(r"^## `([^`]+\.csv)`", line)
        if m:
            cur = m.group(1)
            exp[cur] = []
            continue
        if line.startswith("## "):
            cur = None
        m = re.match(r"^\| `([^`]+)` \|", line)
        if cur and m:
            exp[cur].append(m.group(1))
    return exp


def check(out):
    """Problems found (empty list = good): hashes, schema, splits, JSONL, privacy."""
    out = Path(out)
    p = []
    man = json.loads((out / "manifest.json").read_text())
    for rel, info in man["files"].items():
        f = out / rel
        if not f.exists():
            p.append(f"missing {rel}")
        elif sha(f.read_bytes()) != info["sha256"]:
            p.append(f"hash mismatch {rel}")
    expected = {rel: [c for c, *_ in S[name]] for name, rel in TABLE_PATH.items()}
    if str(man.get("version", VERSION)).split(".")[0] != VERSION.split(".")[0]:
        expected = schema_from_md(out)     # an older release: check it against the schema it shipped with
    for rel, cols_ok in expected.items():
        f = out / rel
        if f.exists():
            cols = list(pd.read_csv(f, nrows=0).columns)
            if cols != cols_ok:
                p.append(f"schema mismatch {rel}")
    fr = pd.read_csv(out / TABLE_PATH["frames"])
    sp = fr.groupby("site_id").split.nunique()
    if (sp > 1).any():
        p.append(f"site in two splits: {list(sp[sp > 1].index)}")
    if fr.frame_id.duplicated().any():
        p.append("duplicate frame_id")
    lo = pd.read_csv(out / TABLE_PATH["loops"])
    if not set(lo.frame_id) <= set(fr.frame_id):
        p.append("loops reference unknown frames")
    for f in sorted((out / "llm").glob("*.jsonl")):
        for i, line in enumerate(open(f)):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                p.append(f"{f.name}:{i + 1} not JSON")
                break
            if f.name.startswith("sft_") and [m["role"] for m in r["messages"]] != ["system", "user", "assistant"]:
                p.append(f"{f.name}:{i + 1} bad roles")
                break
    priv = out / "_private" / "id_map.json"
    if priv.exists():
        secrets = [w for w in json.loads(priv.read_text())["subjects"] if w not in ("unknown",) and len(w) >= 3]
        names = [Path(v).stem for v in json.loads(priv.read_text())["files"].values()]
        for f in sorted(out.rglob("*")):
            if f.is_file() and "_private" not in f.parts and f.suffix in (".csv", ".json", ".jsonl", ".md"):
                t = f.read_text().lower()
                leak = [w for w in secrets if re.search(rf"\b{re.escape(w.lower())}\b", t)]
                leak += [n for n in names[:200] if n.lower() in t]
                if leak:
                    p.append(f"privacy: {f.relative_to(out)} contains {sorted(set(leak))[:3]}")
    for f in sorted(out.rglob("*")):
        if f.is_file() and "_private" not in f.parts and f.suffix in (".csv", ".json", ".jsonl", ".md"):
            if re.search(r"(/Users/|/home/|/sessions/|C:\\\\)", f.read_text()):
                p.append(f"absolute path in {f.relative_to(out)}")
    return p


# ------------------------------------------------------------------------------------------- vision (local only)
def build_vision(out, zips_folder, vision_out):
    """Images from the original zips or image folders + COCO-style keypoint labels (apex) and boxes, for detector training.
    Written to `vision_out`, which must stay out of git (images of real fingers)."""
    from PIL import Image
    out, vo = Path(out), Path(vision_out)
    priv = json.loads((out / "_private" / "id_map.json").read_text())
    want = {orig: fid for fid, orig in priv["files"].items()}
    fr = pd.read_csv(out / TABLE_PATH["frames"])
    lo = pd.read_csv(out / TABLE_PATH["loops"])
    keep = set(fr[fr.measured].frame_id)
    images, anns, found, sizes = [], [], {}, {}
    def sources(folder):
        """(name, bytes-loader) for every image in the folder's zips and image sub-folders."""
        root = Path(folder).expanduser()
        for z in sorted(root.glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                for info in sorted(zf.infolist(), key=lambda i: i.filename):
                    if "__MACOSX" not in info.filename:
                        yield Path(info.filename).name, (lambda zf=zf, info=info: zf.read(info))
        for p in sorted(q for q in root.rglob("*") if q.is_file() and q.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            yield p.name, p.read_bytes

    for base, load in sources(zips_folder):
        if base not in want or want[base] in found or want[base] not in keep:
            continue
        fid = want[base]
        img = Image.open(io.BytesIO(load())).convert("RGB")
        rel = Path(fr.loc[fr.frame_id == fid, "split"].iloc[0]) / (fid.replace("/", "__") + ".png")
        (vo / rel.parent).mkdir(parents=True, exist_ok=True)
        img.save(vo / rel)
        found[fid] = len(images) + 1
        images.append(dict(id=found[fid], file_name=str(rel), width=img.width, height=img.height, frame_id=fid))
        sizes[fid] = (img.width, img.height)
    for k, l in enumerate(lo[lo.frame_id.isin(found)].sort_values("loop_id").itertuples(), 1):
        W, H = sizes[l.frame_id]
        half = (l.total_px / 2) if np.isfinite(l.total_px) else 12.0
        x0, y0 = max(0.0, l.apex_x_px - half), max(0.0, l.apex_y_px - 8)          # box clipped to the frame
        x1, y1 = min(float(W), l.apex_x_px + half), min(float(H), l.apex_y_px + 50)
        anns.append(dict(id=k, image_id=found[l.frame_id], category_id=1, keypoints=[float(l.apex_x_px), float(l.apex_y_px), 2],
                         num_keypoints=1, bbox=[float(x0), float(y0), float(x1 - x0), float(y1 - y0)], area=float((x1 - x0) * (y1 - y0)), iscrowd=0,
                         loop_id=l.loop_id, resolved=bool(l.resolved), auto_label=True))
    coco = dict(info=dict(description="Captrail capillary apex auto-labels (pipeline output, check before training)", version=VERSION),
                categories=[dict(id=1, name="capillary_loop", keypoints=["apex"], skeleton=[])], images=images, annotations=anns)
    (vo / "annotations.json").write_text(json.dumps(coco, indent=1) + "\n")
    print(f"-> {vo}: {len(images)} images, {len(anns)} loop labels (LOCAL ONLY: real fingers, do not commit)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", nargs="?", help="batch_run.py output folder")
    ap.add_argument("--out", default="datasets/captrail-v1")
    ap.add_argument("--compare-with", default=None, help="a second batch folder (same inputs, other machine) for platform stability")
    ap.add_argument("--check", default=None, metavar="DATASET", help="only verify an exported dataset")
    ap.add_argument("--vision-images", default=None, help="folder with the original capture zips")
    ap.add_argument("--vision-out", default="datasets/_local_vision")
    a = ap.parse_args()
    if a.check:
        pr = check(a.check)
        print("OK" if not pr else "\n".join(pr))
        sys.exit(1 if pr else 0)
    if not a.batch:
        ap.error("batch folder required")
    build(a.batch, a.out, a.compare_with)
    if a.vision_images:
        build_vision(a.out, a.vision_images, a.vision_out)
