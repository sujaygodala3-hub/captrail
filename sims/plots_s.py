"""Figures for S1 to S7. Same style as plots.py (one maroon ramp, ordered magnitude)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plots  # noqa: F401  (sets rcParams)
from plots import GRID, INK, MUTED

RAMP5 = ["#e8c4c4", "#d9a0a0", "#b95a5a", "#8c2f39", "#4a0f1c"]


def _mark(ax, scope, label=True):
    ax.axvline(scope.sigma_um, color=MUTED, ls=":", lw=1.2)
    if label:
        ax.text(scope.sigma_um, ax.get_ylim()[1], " scope sigma" + ("" if scope.measured.get("sigma_um") else " (ASSUMED)"),
                color=MUTED, fontsize=8, va="top")


def _banner(fig, scope):
    if scope.assumed():
        fig.text(0.01, 0.005, "Uses ASSUMED scope values for: " + ", ".join(scope.assumed()) + ". Not a result until S0 and S5.",
                 color="#8c2f39", fontsize=7.5)


def fig_s1(df, scope, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    d = df[(df.pixel == scope.um_per_px)]
    for ax, kind, est, title in zip(axes, ["limb", "apical", "total"], ["fwhm", "fwhm", "fwhm"],
                                    ["Limb width (half-depth)", "Apical width", "Total loop width"]):
        x = d[(d.kind == kind) & (d.estimator == est)]
        for c, t in zip(RAMP5[::-1][:x.true.nunique()][::-1], sorted(x.true.unique())):
            g = x[x.true == t].sort_values("sigma")
            ax.plot(g.sigma, g.bias_pct, marker="o", color=c, label=f"{t:g} um")
        ax.axhline(0, color=MUTED, lw=1); ax.axhspan(-10, 10, color=GRID, alpha=0.5, lw=0)
        ax.set_title(title, loc="left", fontsize=10); ax.set_xlabel("PSF sigma (um)"); ax.set_ylim(-20, 150)
        ax.legend(title="true", fontsize=7)
        _mark(ax, scope)
    axes[0].set_ylabel("width bias (%)")
    fig.suptitle("S1. Width bias vs blur at the scope's pixel size (shaded = within 10%)", x=0.01, ha="left", fontsize=11)
    _banner(fig, scope); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "s1_width_bias.png"); plt.close(fig)


def fig_s2(df, scope, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, f in zip(axes, sorted(df.field_mm.unique())):
        x = df[df.field_mm == f]
        cs = sorted(x.contrast.unique())
        for c, ct in zip(RAMP5[-len(cs):], cs):
            g = x[x.contrast == ct].groupby("sigma").err_pct.mean()
            ax.plot(g.index, g.values, marker="o", color=c, label=f"contrast {int(ct * 100)}%")
        ax.axhline(0, color=MUTED, lw=1); ax.set_title(f"{f:g} mm field", loc="left", fontsize=10)
        ax.set_xlabel("PSF sigma (um)")
    axes[0].set_ylabel("mean density error (%)"); axes[0].legend(fontsize=8)
    fig.suptitle("S2. Apex-count density error (true 3 to 12 per mm, skin texture on)", x=0.01, ha="left", fontsize=11)
    _banner(fig, scope); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "s2_density.png"); plt.close(fig)


def fig_s3(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for cond, c in (("no_oil", RAMP5[1]), ("oil", RAMP5[4])):
        g = df[df.condition == cond]
        axes[0].plot(g.depth_um, g.contrast, marker="o", color=c, label=cond)
        axes[1].plot(g.depth_um, g.limb_bias_pct, marker="o", color=c, label=cond)
    axes[0].set_ylabel("loop contrast"); axes[1].set_ylabel("limb width bias (%)")
    for a in axes:
        a.set_xlabel("loop depth (um)"); a.legend()
    fig.suptitle("S3. Depth and oil (PHENOMENOLOGICAL model, fitted to your captures)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "s3_depth_oil.png"); plt.close(fig)


def fig_s4(df, scope, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.sigma, df.p_45_called_giant * 100, marker="o", color=RAMP5[2], label="45 um read as 50 or more")
    ax.plot(df.sigma, df.p_55_called_not_giant * 100, marker="s", color=RAMP5[4], label="55 um read as under 50")
    ax.axhline(5, color=MUTED, ls="--", lw=1); ax.set_ylabel("misclassification (%)"); ax.set_xlabel("PSF sigma (um)")
    ax.set_title("S4. Calls at the 50 um EULAR giant threshold", loc="left", fontsize=10); ax.legend(fontsize=8)
    _mark(ax, scope); _banner(fig, scope); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "s4_misclassification.png"); plt.close(fig)


def fig_s6(df, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["resistance", "flow", "velocity", "pressure"]
    ws = sorted(df.true_width.unique())
    for i, (w, c) in enumerate(zip(ws, RAMP5[-len(ws):])):
        v = [df[(df.true_width == w) & (df.assumption == n)]["median"].iloc[0] for n in names]
        ax.bar(np.arange(4) + i * 0.8 / len(ws), v, width=0.8 / len(ws), color=c, label=f"{w:g} um")
    ax.set_xticks(np.arange(4) + 0.4); ax.set_xticklabels(["resistance\n(r^-4)", "shear, fixed flow\n(r^-3)", "shear, fixed v\n(r^-1)", "shear, fixed dP\n(r^+1)"], fontsize=8)
    ax.axhline(0, color=MUTED, lw=1); ax.set_ylabel("median error (%)"); ax.legend(title="true limb", fontsize=8)
    ax.set_title("S6. Width error amplified (idealized Poiseuille tube, not a flow prediction)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "s6_propagation.png"); plt.close(fig)


def fig_s7(res, out):
    m = res["matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for j in range(m.shape[1]):
        ax.scatter(m.mean(1), m[:, j], s=14, color=RAMP5[1 + j % 4], alpha=0.8)
    ax.set_xlabel("subject mean apical width (um)"); ax.set_ylabel("single capture (um)")
    ax.set_title(f"S7. Simulated re-seating: ICC {res['icc']:.2f} (95% {res['ci'][0]:.2f} to {res['ci'][1]:.2f})", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "s7_repeatability.png"); plt.close(fig)
