"""S1 to S7 from measured scope parameters.

    python run_sims.py all --scope results/scope_params.json            # full run
    python run_sims.py all --quick                                       # smoke run on assumed defaults
    python run_sims.py s1 s6 --scope results/scope_params.json
    python run_sims.py s8 sens           # model-misspecification check and scope-uncertainty sensitivity

S5 and the real-data parts of S3 and S7 read optional CSVs (data/README.md). Outputs go to
results/<tag>/ where tag is 'measured' only when every scope value came from S0.
All numbers are model outputs until S5 passes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capsim import intake, stats, sweeps as W
from capsim.scope import ScopeParams
import plots_s as PS

ROOT = Path(__file__).parent


def main(steps, scope_path, quick, data, out_dir=None, look_path=None):
    sc = ScopeParams.load(scope_path)
    out = Path(out_dir) if out_dir else ROOT / "results" / sc.tag()
    out.mkdir(parents=True, exist_ok=True)
    sc.save(out / "scope_used.json")
    data = Path(data)
    summary = dict(scope_tag=sc.tag(), assumed=sc.assumed(), quick=quick)
    want = lambda s: "all" in steps or s in steps

    if want("s1"):
        df = W.s1_width_bias(sc, quick); df.to_csv(out / "s1_width_bias.csv", index=False)
        PS.fig_s1(df, sc, out)
        summary["s1_min_limb_under_10pct"] = W.s1_min_resolvable_limb(df, round(sc.sigma_um, 2), sc.um_per_px)
    if want("s2"):
        df = W.s2_density(sc, quick); df.to_csv(out / "s2_density.csv", index=False)
        W.s2_texture(sc, quick).to_csv(out / "s2_texture.csv", index=False)
        PS.fig_s2(df, sc, out)
        summary["s2_auc_reduced_density"] = {f"contrast{c}": W.s2_roc_reduced_ci(df, sc.sigma_um, c) for c in sorted(df.contrast.unique())}
    if want("s3"):
        pairs = data / "s3" / "oil_pairs.csv"
        if pairs.exists():
            d = intake.read_csv(pairs, ["condition", "contrast", "sigma_um"])
            depth, c0 = float(json.loads((ROOT / "configs" / "s3_config.json").read_text())["assumed_depth_um"]), 0.5
            fit = {c: W.fit_depth_params(g.contrast.mean(), g.sigma_um.mean(), c0, sc.sigma_um, depth) for c, g in d.groupby("condition")}
            p_no, p_oil = dict(zip(("mu", "k"), fit["no_oil"])), dict(zip(("mu", "k"), fit["oil"]))
            summary["s3_fit"] = dict(no_oil=p_no, oil=p_oil, assumed_depth_um=depth, c0=c0)
        else:
            p_no, p_oil = dict(mu=0.010, k=0.020), dict(mu=0.004, k=0.008)
            summary["s3_fit"] = "ILLUSTRATIVE ONLY: no data/s3/oil_pairs.csv, mu and k are placeholders"
        df = W.s3_depth_oil(sc, p_no, p_oil, quick=quick); df.to_csv(out / "s3_depth_oil.csv", index=False)
        PS.fig_s3(df, out)
    if want("s4"):
        df = W.s4_misclassification(sc, quick); df.to_csv(out / "s4_misclassification.csv", index=False)
        PS.fig_s4(df, sc, out)
        summary["s4_max_sigma_err_le_5pct"] = W.max_sigma_for_error(df)
        summary["s4_density_call"] = W.s4_density_call(sc, quick)
    if want("s5"):
        prof = data / "s5" / "slide_profile.csv"
        s5 = {}
        if prof.exists():
            real = pd.read_csv(prof).iloc[:, 0].values
            cfg0 = json.loads((ROOT / "configs" / "s0_config.json").read_text())
            tick = cfg0.get("tick_width_um") or 2.0
            from capsim import calibration as C
            _, fit = C.sigma_from_ticks(real, sc.um_per_px, cfg0.get("pitch_um", 10.0), tick)    # ruler contrast from the real profile
            _, sim = W.slide_profile_sim(sc, pitch_um=cfg0.get("pitch_um", 10.0), tick_um=tick, contrast=float(fit["amp"] / fit["bg"]))
            s5["slide_profile"] = W.compare_profiles(real, sim)
        man = data / "s5" / "manual_widths.csv"
        if man.exists():
            m = intake.read_csv(man, ["rater1_um", "rater2_um", "tool_um"])
            s5["agreement"] = W.s5_agreement(m)
            s5["sim_vs_real"] = W.s5_sim_bias_vs_real(sc, m, quick)
        s5["status"] = "no S5 data found; S1 to S4 are NOT validated" if not s5 else "see numbers; fix the model if sim_vs_real.passes is False"
        summary["s5"] = s5
    if want("s6"):
        df = W.s6_propagation(sc, quick=quick); df.to_csv(out / "s6_propagation.csv", index=False)
        W.s6_shear_table().to_csv(out / "s6_shear_table.csv", index=False)
        PS.fig_s6(df, out)
    if want("s7"):
        r = W.s7_repeatability(sc, quick)
        PS.fig_s7(r, out)
        summary["s7"] = dict(icc=r["icc"], ci=r["ci"], cv_pct=r["cv_pct"], n=r["n"])
        rep = data / "s7" / "repeat_widths.csv"
        if rep.exists():
            summary["s7_real"] = stats.repeatability(pd.read_csv(rep)["width_um"].values)
    if want("s8"):
        from capsim import misspec
        df = misspec.misspecification(sc, reps=4 if quick else 15)
        df.to_csv(out / "s8_misspecification.csv", index=False)
        summary["s8_max_abs_limb_bias_gap_pct"] = float(np.nanmax(np.abs(df.limb_bias_gap))) if df.limb_bias_gap.notna().any() else None
        summary["s8_unresolvable_ruler_cases"] = int(df.sigma_gauss_fit.isna().sum())
    if want("sens"):
        from capsim import sensitivity as SE
        summary["sens_tornado_limb12"] = SE.tornado(sc, "limb12_bias_pct")
        summary["sens_tornado_giant_err"] = SE.tornado(sc, "giant_call_error")
        for k in ("limb12_bias_pct", "giant_call_error"):
            mc = SE.monte_carlo(sc, k, n=15 if quick else 60)
            summary[f"sens_mc_{k}"] = {a: b for a, b in mc.items() if a != "values"}
    if want("s9"):
        from PIL import Image
        from capsim import domain_gap as DG, render_rgb as R
        lf = Path(look_path) if look_path else ROOT / "configs" / "look_fitted.json"
        look = (R.Look.celestron() if look_path == "celestron" else R.Look.load(lf)) if (look_path == "celestron" or lf.exists()) else R.Look.dataset()
        look = __import__("dataclasses").replace(look, px_um=sc.um_per_px) if sc.measured.get("um_per_px") else look
        rng = np.random.default_rng(9)
        frames = [R.render_scene(rng, look)["img"] for _ in range(24 if quick else 60)]
        Image.fromarray(np.concatenate([np.concatenate(frames[r * 4:(r + 1) * 4], 1) for r in range(4)], 0)).resize((1280, 960)).save(out / "fig9_synthetic_examples.png")
        s9 = dict(look=str(look_path or ("configs/look_fitted.json" if lf.exists() else "Look.dataset() defaults (NOT fitted)")), n_synth=len(frames))
        real_dir = data / "real_images"
        if real_dir.exists() and any(p.suffix.lower() in intake.IMG_EXT for p in real_dir.rglob("*")):
            gap = DG.compare_dirs(real_dir, frames, max_real=300)
            s9.update(auc=gap["auc"], auc_sd=gap["auc_sd"], n_real=gap["n_real"], worst_features=sorted(gap["features"], key=lambda r: -abs(r["smd"]))[:5])
        else:
            s9["note"] = "no data/real_images: domain gap not measured"
        summary["s9"] = s9
    from capsim import gates, report
    summary["scope_sigma_um"] = sc.sigma_um
    summary["s0_report"] = sc.notes.get("s0_report", {})
    rows = gates.evaluate(summary)
    summary["gates"], summary["validated"] = rows, gates.validated(rows)
    report.write_report(out, summary, sc, rows, report.manifest(sc, quick, ROOT))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", help="s1 .. s9, sens, or all")
    ap.add_argument("--scope", default="results/scope_params.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="output folder (default results/<scope tag>)")
    ap.add_argument("--look", default=None, help="S9: a Look json from fit_look.py, or 'celestron'")
    a = ap.parse_args()
    main(a.steps, a.scope, a.quick, a.data, a.out, a.look)
