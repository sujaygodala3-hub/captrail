"""Pass/fail gates from ANALYSIS_PLAN.md, evaluated on a run's summary dict.

Thresholds are fixed constants here so changing one is a visible code change."""
from __future__ import annotations

import numpy as np

MIN_TICKS = 8
UMPX_CV_MAX_PCT = 1.5
PT_R2_MIN = 0.98
WIDTH_TOL_PCT = 10.0
GIANT_ERR_MAX = 0.05
ICC_RATERS_MIN = 0.85
SD_RATIO = (0.5, 2.0)
SLIDE_NRMSE_MAX = 0.20
MIN_CAPILLARIES = 10
GAP_AUC_MAX = 0.75          # D1: a classifier should not separate synthetic from held-out real frames much better than this


def _row(name, status, detail):
    return dict(gate=name, status=status, detail=detail)


def evaluate(summary: dict, s0_report: dict | None = None):
    """Return a list of gate rows. Status is PASS, FAIL or NOT RUN (no data)."""
    rows = []
    s0 = s0_report or summary.get("s0_report") or {}
    if "um_per_px" in s0:
        cv = s0["um_per_px"].get("cv_pct")
        rows.append(_row("S0 um/px repeatability", "PASS" if cv is not None and cv <= UMPX_CV_MAX_PCT else "FAIL",
                         f"CV {cv:.2f}% (limit {UMPX_CV_MAX_PCT}%), n={s0['um_per_px'].get('n')}"))
        if s0.get("slide_skipped"):
            rows.append(_row("S0 ruler ticks found", "FAIL", f"{len(s0['slide_skipped'])} capture(s) skipped"))
    else:
        rows.append(_row("S0 um/px repeatability", "NOT RUN", "no slide captures"))
    pt = s0.get("photon_transfer")
    if pt:
        ok = pt.get("linear") and pt.get("r2", 0) >= PT_R2_MIN
        rows.append(_row("S0 photon transfer linear", "PASS" if ok else "FAIL", f"r2 {pt.get('r2'):.3f}, gain {pt.get('g'):.3f}"))
    else:
        rows.append(_row("S0 photon transfer linear", "NOT RUN", "no blank-card frames"))
    rows.append(_row("All scope values measured", "PASS" if not summary.get("assumed") else "FAIL",
                     "none assumed" if not summary.get("assumed") else "still ASSUMED: " + ", ".join(summary["assumed"])))

    ms = summary.get("s4_max_sigma_err_le_5pct")
    sig = summary.get("scope_sigma_um")
    if isinstance(ms, dict) and sig is not None:
        m, reached = ms.get("max_sigma_um"), ms.get("limit_reached")
        ok = m is not None and sig <= m
        note = "" if reached else " (no failure found up to the top of the sweep, so the true limit is higher)"
        rows.append(_row("S4 giant-call error at scope sigma", "PASS" if ok else "FAIL",
                         f"largest sigma with error <= {GIANT_ERR_MAX:.0%} is {m} um{note}; scope sigma {sig:.2f} um"))

    s5 = summary.get("s5", {})
    ag = s5.get("agreement")
    if ag:
        rows.append(_row("S5 inter-rater ICC", "PASS" if ag["icc_raters"] >= ICC_RATERS_MIN else "FAIL",
                         f"ICC {ag['icc_raters']:.2f} (limit {ICC_RATERS_MIN})"))
        n = ag["tool_vs_raters"]["n"]
        rows.append(_row("S5 enough capillaries", "PASS" if n >= MIN_CAPILLARIES else "FAIL", f"n={n} (need {MIN_CAPILLARIES})"))
    sv = s5.get("sim_vs_real")
    if sv:
        rows.append(_row("S5 simulated bias consistent with real mean bias", "PASS" if sv["passes"] else "FAIL",
                         f"sim bias {sv['sim_mean_bias_um']:.2f} um; real bias {sv['real']['bias']:.2f} um (SD {sv['real']['sd']:.2f}, n {sv['real']['n']})"))
        if "sd_ratio" in sv:
            ok = SD_RATIO[0] <= sv["sd_ratio"] <= SD_RATIO[1]
            rows.append(_row("S5 simulated SD within 0.5 to 2x real", "PASS" if ok else "FAIL", f"sim/real SD ratio {sv['sd_ratio']:.2f}"))
    sp = s5.get("slide_profile")
    if sp:
        rows.append(_row("S5 slide profile match", "PASS" if sp["nrmse"] <= SLIDE_NRMSE_MAX else "FAIL",
                         f"NRMSE {sp['nrmse']:.3f} (limit {SLIDE_NRMSE_MAX})"))
    s9 = summary.get("s9")
    if s9:
        if "auc" in s9:
            rows.append(_row("D1 synthetic vs real image gap", "PASS" if s9["auc"] <= GAP_AUC_MAX else "FAIL",
                             f"two-sample AUC {s9['auc']:.2f} (limit {GAP_AUC_MAX}; 0.5 = indistinguishable), n_real {s9['n_real']}"))
        else:
            rows.append(_row("D1 synthetic vs real image gap", "NOT RUN", s9.get("note", "no real images")))
    if not s5 or ("agreement" not in s5 and "slide_profile" not in s5):
        rows.append(_row("S5 forward-model validation", "NOT RUN", "no S5 data: S1 to S4 are NOT validated"))
    return rows


def validated(rows):
    """True only if every S5 gate ran and passed."""
    s5 = [r for r in rows if r["gate"].startswith("S5")]
    return bool(s5) and all(r["status"] == "PASS" for r in s5)
