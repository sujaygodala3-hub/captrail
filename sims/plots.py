"""Figures for the Capillary core experiments. One sequential maroon ramp = true width (ordered magnitude)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# single-hue ramp, light -> dark, ordered by true width (8, 12, 20, 40 um)
RAMP = {8: "#d9a0a0", 12: "#b95a5a", 20: "#8c2f39", 40: "#4a0f1c"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 140,
})


def _band(ax, df, x, y, w, ls="-", marker="o", scale=100, baseline=None, **kw):
    g = df[df.width == w].groupby(x)[y]
    m, s = g.median(), g.std()
    ax.plot(m.index, m.values, ls=ls, marker=marker, color=RAMP[w], **kw)
    ax.fill_between(m.index, m - s, m + s, color=RAMP[w], alpha=0.12, lw=0)


def _legend_widths(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, lw=2, label=f"{w} um") for w, c in RAMP.items()]
    ax.legend(handles=h, title="true width", loc="best")


def _pct(df, col):
    d = df.copy()
    d[col] = (d[col] / d["width"] - 1) * 100
    return d


def fig_blur(e1, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ["fwhm", "otsu"], ["Half-depth width (FWHM) estimator", "Naive Otsu-area estimator"]):
        d = _pct(e1, col)
        for w in RAMP:
            _band(ax, d, "sigma", col, w)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("PSF blur sigma (um)")
    axes[0].set_ylabel("width error (%)")
    axes[0].set_ylim(-30, 400)
    _legend_widths(axes[1])
    fig.suptitle("E1. Blur inflates width of thin vessels first (median, band = 1 SD over replicates)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig1_blur_vs_width_error.png"); plt.close(fig)


def fig_noise(e2, out):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    d = e2.copy(); d["err"] = (d["fwhm"] / d["width"] - 1) * 100
    for w in RAMP:
        g = d[d.width == w].groupby("photons")["err"]
        ax.plot(g.std().index, g.std().values, marker="o", color=RAMP[w])
    ax.set_xscale("log"); ax.set_xlabel("photons per pixel at full brightness")
    ax.set_ylabel("SD of width error (%)")
    ax.set_title("E2. Shot noise: spread of width estimates", loc="left", fontsize=10)
    _legend_widths(ax)
    fig.tight_layout(); fig.savefig(out / "fig2_noise.png"); plt.close(fig)


def fig_illum(e3, out):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    d = _pct(e3.assign(width=12), "fwhm")
    for flat, ls, lab in [(True, "-", "flat-fielded"), (False, "--", "raw")]:
        g = d[d.flatfield == flat].groupby("illum")["fwhm"]
        ax.plot(g.median().index * 100, g.median().values, ls=ls, marker="o", color=RAMP[20], label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("illumination variation across field (%)"); ax.set_ylabel("width error (%), 12 um vessel")
    ax.set_title("E3. Lighting gradient, with and without correction", loc="left", fontsize=10)
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig3_illumination.png"); plt.close(fig)


def fig_pixel(e4, out):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    d = _pct(e4, "fwhm")
    for w in RAMP:
        _band(ax, d, "pixel", "fwhm", w)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("detector pixel size (um/px)"); ax.set_ylabel("width error (%)")
    ax.set_title("E4. Pixel size (blur sigma fixed at 1.5 um)", loc="left", fontsize=10)
    _legend_widths(ax)
    fig.tight_layout(); fig.savefig(out / "fig4_pixel_size.png"); plt.close(fig)


def fig_count(e5, out):
    fig, ax = plt.subplots(figsize=(5.4, 4))
    d = e5.assign(err=(e5.counted - e5.true) / e5.true * 100)
    ramp = ["#d9a0a0", "#b95a5a", "#8c2f39", "#4a0f1c"]
    for c, ct in zip(ramp[::-1], sorted(d.contrast.unique())):
        g = d[d.contrast == ct].groupby("sigma")["err"].mean()
        ax.plot(g.index, g.values, marker="o", color=c, label=f"contrast {int(ct * 100)}%")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("PSF blur sigma (um)"); ax.set_ylabel("mean count error (%), 7 true loops")
    ax.set_title("E5. Loop counting vs blur and vessel contrast", loc="left", fontsize=10)
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig5_loop_count.png"); plt.close(fig)


def fig_prop(e6, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for w in RAMP:
        g = e6[e6.width == w].groupby("sigma")
        axes[0].plot(g["w_err_pct"].median().index, g["w_err_pct"].median().values, marker="o", color=RAMP[w])
        axes[1].plot(g["R_err_pct"].median().index, g["R_err_pct"].median().values, marker="o", color=RAMP[w])
    axes[0].set_ylabel("width error (%)"); axes[1].set_ylabel("flow-resistance error (%)")
    axes[1].set_yscale("symlog", linthresh=10)
    for a in axes:
        a.axhline(0, color=MUTED, lw=1); a.set_xlabel("PSF blur sigma (um)")
    axes[0].set_title("Measured width error", loc="left", fontsize=10)
    axes[1].set_title("Same error through Poiseuille R ~ r^-4", loc="left", fontsize=10)
    _legend_widths(axes[0])
    fig.suptitle("E6. Small width errors become roughly 4x larger resistance errors (idealized straight tube)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig6_error_propagation.png"); plt.close(fig)


def fig_example(out):
    from capsim import geometry as G, optics as O
    sp = G.LoopSpec(cx=80, y_base=10, limb_gap=50, limb_len=150, width_art=12, width_ven=16, apex_bulge=10)
    mask = G.render_mask([sp], (160, 220), 0.5)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    axes[0].imshow(mask, cmap="gray_r"); axes[0].set_title("ground truth", fontsize=9)
    for ax, s, ph in zip(axes[1:], [1, 3, 6], [20000, 2000, 500]):
        img = O.forward(mask, 0.5, sigma_um=s, photons=ph, read_noise=0.005, rng=np.random.default_rng(3))
        ax.imshow(img, cmap="gray", vmin=0.35, vmax=0.85); ax.set_title(f"blur {s} um, {ph} photons", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "fig0_example_images.png"); plt.close(fig)


def make_all(e1, e2, e3, e4, e5, e6, out):
    fig_example(out); fig_blur(e1, out); fig_noise(e2, out); fig_illum(e3, out)
    fig_pixel(e4, out); fig_count(e5, out); fig_prop(e6, out)
