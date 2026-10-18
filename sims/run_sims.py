"""S1 to S7 from measured scope parameters.

    python run_sims.py all --scope results/scope_params.json            # full run
    python run_sims.py all --quick                                       # smoke run on assumed defaults
    python run_sims.py s1 s6 --scope results/scope_params.json

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


def main(steps, scope_path, quick, data):
    sc = ScopeParams.load(scope_path)
    out = ROOT / "results" / sc.tag()
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
        summary["s2_auc_reduced_density"] = {f"contrast{c}": W.s2_roc_reduced(df, sc.sigma_um, c) for c in sorted(df.contrast.unique())}
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
            _, sim = W.slide_profile_sim(sc, tick_um=json.loads((ROOT / "configs" / "s0_config.json").read_text()).get("tick_width_um") or 2.0)
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", help="s1 .. s7 or all")
    ap.add_argument("--scope", default="results/scope_params.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    main(a.steps, a.scope, a.quick, a.data)
