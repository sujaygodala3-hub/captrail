"""Build results/<tag>/REPORT.md: one page with the scope, the gates, and the headline numbers."""
from __future__ import annotations

import json
import platform
from pathlib import Path

import pandas as pd

from . import gates as G


def manifest(scope, quick, repo_root):
    import numpy, scipy, skimage
    git = "unknown"
    try:
        import subprocess
        git = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        pass
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                skimage=skimage.__version__, git_commit=git, quick=quick)


def _md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def write_report(out: Path, summary: dict, scope, rows, mf):
    ok = G.validated(rows)
    L = []
    L.append(f"# Captrail simulation report ({summary['scope_tag']})\n")
    L.append("**" + ("S5 gate passed: S1 to S4 may be reported as validated against the real scope." if ok else
             "MODEL NOT VALIDATED: every number below is a model output. Do not report S1 to S4 as results about the Celestron.") + "**\n")
    if mf.get("quick"):
        L.append("_Quick run: few replicates, not for reporting._\n")
    L.append("## Scope values\n")
    rows_sc = []
    for k in ("um_per_px", "sigma_um", "photons", "read_noise", "illum_strength", "illum_angle_deg", "contrast", "contrast_oil"):
        v = getattr(scope, k)
        rows_sc.append(dict(value=k, number=("n/a" if v is None else f"{v:.4g}"),
                            source="MEASURED" if scope.measured.get(k) else "ASSUMED",
                            sd=f"{scope.uncertainty[k]:.3g}" if k in scope.uncertainty else ""))
    L.append(_md_table(pd.DataFrame(rows_sc)) + "\n")
    L.append("## Gates (ANALYSIS_PLAN.md)\n")
    L.append(_md_table(pd.DataFrame(rows)) + "\n")
    for key, title in (("s1_min_limb_under_10pct", "Smallest limb width with |bias| <= 10% at the scope sigma (um)"),
                       ("s4_max_sigma_err_le_5pct", "Largest sigma keeping the 45 vs 55 um giant call error <= 5% (um)"),
                       ("s8_max_abs_limb_bias_gap_pct", "S8: largest limb-bias gap, Gaussian model vs non-Gaussian truth (percentage points)")):
        if key in summary:
            L.append(f"- {title}: **{summary[key]}**")
    for key in ("s2_auc_reduced_density", "s4_density_call", "s7", "s7_real"):
        if key in summary:
            L.append(f"- {key}: `{json.dumps(summary[key], default=float)}`")
    for key in ("sens_mc_limb12_bias_pct", "sens_mc_giant_call_error"):
        if key in summary:
            d = summary[key]
            L.append(f"- Uncertainty ({key}): mean {d['mean']:.3g}, 95% interval {d['lo95']:.3g} to {d['hi95']:.3g} over scope uncertainty")
    if "sens_tornado_limb12" in summary:
        L.append("\n### What moves the 12 um limb bias most (+-25% one at a time)\n")
        t = pd.DataFrame(summary["sens_tornado_limb12"])[["param", "measured", "swing"]]
        L.append(_md_table(t))
    L.append("\n## Reproducibility\n")
    L.append("```\n" + json.dumps(mf, indent=2) + "\n```\n")
    (out / "REPORT.md").write_text("\n".join(L))
