"""S1 to S7: the experiments, written to consume measured scope parameters (capsim.scope).

Every function returns a pandas DataFrame (or dict) and takes `quick=True` for a short run.
Single-loop images are 160 x 220 um; distal-row images are n*spacing wide.
All numbers are model outputs until S5 passes (see README).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import geometry as G, measure as M, optics as O, propagate as P, stats as S
from .scope import ScopeParams

LIMB_Y = (25.0, 75.0)          # um above the bottom of the field: straight-limb ROI
FOV_H = 220.0


def _fov(px, w=160.0, h=FOV_H):
    return math.ceil(w / px) * px, math.ceil(h / px) * px


def _roi(px, fov_h):
    return int((fov_h - LIMB_Y[1]) / px), int((fov_h - LIMB_Y[0]) / px)


def single_loop_image(scope: ScopeParams, spec_kw, rng, sigma=None, oil=False, **over):
    """One loop through the forward model. Returns (img, px, loop_cx_um, roi_rows, spec)."""
    px, res = scope.um_per_px, scope.fine_res()
    over = dict(over)
    if sigma is not None:
        over["sigma_um"] = sigma
    pixel = over.pop("pixel_um", px)
    res = pixel / math.ceil(pixel / 0.5 - 1e-9)
    kw = scope.forward_kwargs(oil=oil, **over)
    kw["pixel_um"] = pixel
    if "photons" not in over and scope.photons is not None:      # None = noiseless
        kw["photons"] = scope.photons * (pixel / scope.um_per_px) ** 2   # photons collected scale with pixel area
    W, H = _fov(pixel)
    spec = G.make_loop(**spec_kw)
    spec.cx, spec.y_base = W / 2, 10.0
    mask = G.render_mask([spec], (W, H), res)
    img = O.forward(mask, res, rng=rng, **kw)
    return img, pixel, spec.cx, _roi(pixel, H), spec


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if (~np.isnan(x)).any() else float("nan")


def _summ(vals, true):
    v = np.asarray(vals, float)
    n_ok = int((~np.isnan(v)).sum())
    if n_ok == 0:
        return dict(mean=np.nan, sd=np.nan, bias_pct=np.nan, n_ok=0)
    m, lo, hi, _ = S.mean_ci(v)
    return dict(mean=m, sd=np.nanstd(v, ddof=1) if n_ok > 1 else np.nan, bias_pct=100 * (m / true - 1), n_ok=n_ok,
                bias_lo95=100 * (lo / true - 1), bias_hi95=100 * (hi / true - 1))


# ----------------------------------------------------------------------------- S1
def s1_width_bias(scope, quick=False, seed=1):
    """Limb (3 estimators), apical and total width bias vs PSF sigma and pixel size."""
    reps = 3 if quick else 12
    sigmas = sorted(set([0.5, 1, 2, 3, 4, 6, 8] + [round(scope.sigma_um, 2)]))
    pixels = sorted(set([scope.um_per_px, 2.0, 4.0])) if quick else sorted(set([scope.um_per_px, 1.0, 2.0, 3.0, 4.0]))
    limbs = [8, 12, 20] if quick else G.LIMB_WIDTHS
    apicals = [16, 30, 45, 55] if quick else G.APICAL_WIDTHS
    totals = [30, 45] if quick else G.TOTAL_WIDTHS
    rows = []
    rng = np.random.default_rng(seed)
    base = dict(limb_len=100.0)
    for pixel in pixels:
        for sg in sigmas:
            def run(kind, true, spec_kw, fns):
                got = {k: [] for k in fns}
                for _ in range(reps):
                    img, px, cx, roi, _ = single_loop_image(scope, spec_kw, rng, sigma=sg, pixel_um=pixel)
                    for k, f in fns.items():
                        got[k].append(f(img, px, cx, roi))
                for k, v in got.items():
                    rows.append(dict(kind=kind, true=true, sigma=sg, pixel=pixel, estimator=k, **_summ(v, true)))
            for w in limbs:
                run("limb", w, dict(limb_um=w, apical_um=w, total_um=w + 28, **base),
                    {"fwhm": lambda i, p, c, r: _nanmean(M.limb_widths_fwhm(i, p, r)),
                     "fixed": lambda i, p, c, r: _nanmean(M.limb_widths_fixed(i, p, r)),
                     "otsu": lambda i, p, c, r: _nanmean(M.limb_widths_otsu(i, p, r))})
            for a in apicals:
                run("apical", a, dict(limb_um=12, apical_um=a, total_um=40, **base),
                    {"fwhm": lambda i, p, c, r: M.apical_width_fwhm(i, p, c)})
            for t in totals:
                run("total", t, dict(limb_um=12, apical_um=12, total_um=t, **base),
                    {"fwhm": lambda i, p, c, r: M.total_width_fwhm(i, p, r)})
    return pd.DataFrame(rows)


def s1_min_resolvable_limb(df, sigma, pixel, tol_pct=10.0, estimator="fwhm"):
    """Smallest limb width whose |bias| stays under tol at the scope's sigma and pixel size."""
    d = df[(df.kind == "limb") & (df.estimator == estimator) & (np.isclose(df.sigma, sigma)) & (np.isclose(df.pixel, pixel))]
    ok = d[d.bias_pct.abs() <= tol_pct].sort_values("true")
    return None if ok.empty else float(ok.true.iloc[0])


# ----------------------------------------------------------------------------- S2
def row_image(scope, density, rng, field_mm=1.0, sigma=None, contrast=None, texture=0.0, oil=False):
    n = max(1, int(round(density * field_mm)))
    spacing = 1000.0 / density
    pixel = scope.um_per_px
    res = pixel / math.ceil(pixel / 0.5 - 1e-9)
    specs, W = G.loop_row(n, spacing, rng, limb_gap=40, limb_len=100, width_art=12, width_ven=12)
    Wf, H = _fov(pixel, W, FOV_H)
    mask = G.render_mask(specs, (Wf, H), res)
    over = {}
    if sigma is not None:
        over["sigma_um"] = sigma
    if contrast is not None:
        over["contrast"] = contrast
    kw = scope.forward_kwargs(oil=oil, **over)
    img = O.forward(mask, res, rng=rng, texture_amp=texture, **kw)
    return img[:, : int(round(W / pixel))], pixel, n


def s2_density(scope, quick=False, seed=2, texture=0.05):
    reps = 3 if quick else 8
    dens = [3, 5, 7, 9, 12]
    sigmas = sorted(set([scope.sigma_um, 2, 4, 6, 8]))
    contrasts = [0.05, 0.10, 0.20] if not quick else [0.10, 0.20]
    fields = [1.0, 2.0] if not quick else [1.0]
    rows, rng = [], np.random.default_rng(seed)
    for f in fields:
        for sg in sigmas:
            for c in contrasts:
                for d in dens:
                    for r in range(reps):
                        img, px, n = row_image(scope, d, rng, f, sg, c, texture)
                        est = M.density_per_mm(img, px)
                        rows.append(dict(field_mm=f, sigma=sg, contrast=c, true_density=d, rep=r,
                                         est_density=est, err_pct=100 * (est / d - 1)))
    return pd.DataFrame(rows)


def s2_texture(scope, quick=False, seed=22):
    """Why density fails first: skin texture amplitude vs count error at the scope's sigma."""
    reps = 3 if quick else 8
    rows, rng = [], np.random.default_rng(seed)
    for amp in [0.0, 0.03, 0.06, 0.10]:
        for d in [5, 7, 9]:
            for r in range(reps):
                img, px, n = row_image(scope, d, rng, 1.0, texture=amp)
                rows.append(dict(texture=amp, true_density=d, rep=r, est_density=M.density_per_mm(img, px)))
    return pd.DataFrame(rows)


def s2_roc_reduced(df, sigma, contrast, field_mm=1.0, thr=7.0):
    """ROC area for 'reduced density' (true < 7/mm) using -estimated density as the score."""
    return s2_roc_reduced_ci(df, sigma, contrast, field_mm, thr)["auc"]


def s2_roc_reduced_ci(df, sigma, contrast, field_mm=1.0, thr=7.0, n_boot=500):
    """AUC for 'reduced density' using only the two true densities adjacent to the threshold (e.g. 5 vs 7),
    so a non-monotone estimator at far densities cannot pull the AUC below 0.5, with a bootstrap 95% interval."""
    d = df[np.isclose(df.sigma, sigma) & np.isclose(df.contrast, contrast) & np.isclose(df.field_mm, field_mm)]
    below, above = d.true_density[d.true_density < thr], d.true_density[d.true_density >= thr]
    if below.empty or above.empty:
        return dict(auc=float("nan"), lo=float("nan"), hi=float("nan"), pair=None)
    lo_d, hi_d = below.max(), above.min()
    d = d[d.true_density.isin([lo_d, hi_d])]
    s, y = -d.est_density.values, (d.true_density < thr).astype(int).values
    lo, hi = S.roc_auc_ci(s, y, n_boot=n_boot)
    return dict(auc=S.roc_auc(s, y), lo=lo, hi=hi, pair=[float(lo_d), float(hi_d)])


# ----------------------------------------------------------------------------- S3
def fit_depth_params(c_obs, sigma_obs, c0, sigma0, depth_um):
    """Solve mu, k for the phenomenological depth model from one condition's measured contrast
    and sigma at an assumed depth. c0 and depth are ASSUMPTIONS (flag them in the write-up)."""
    mu = -math.log(max(c_obs, 1e-6) / c0) / depth_um
    k = math.sqrt(max(sigma_obs ** 2 - sigma0 ** 2, 0.0)) / depth_um
    return mu, k


def s3_depth_oil(scope, params_no_oil, params_oil, c0=0.5, quick=False, seed=3):
    """Bias of apical and limb width, and density count, vs loop depth, with and without oil.

    params_*: dict(mu=, k=) per 'um of depth' (fit with fit_depth_params to real captures).
    PHENOMENOLOGICAL: depth, mu and k are fitted to your captures, not literature values.
    """
    reps = 3 if quick else 10
    rows, rng = [], np.random.default_rng(seed)
    for cond, pr in (("no_oil", params_no_oil), ("oil", params_oil)):
        for depth in ([0, 40, 80, 120] if quick else [0, 20, 40, 60, 80, 100, 120, 150]):
            c, sg = O.depth_model(c0, scope.sigma_um, depth, pr["mu"], pr["k"])
            la, ap = [], []
            for _ in range(reps):
                img, px, cx, roi, _ = single_loop_image(scope, dict(limb_um=12, apical_um=30, total_um=40, limb_len=100.0),
                                                        rng, sigma=sg, contrast=c)
                la.append(_nanmean(M.limb_widths_fwhm(img, px, roi)))
                ap.append(M.apical_width_fwhm(img, px, cx))
            rows.append(dict(condition=cond, depth_um=depth, contrast=c, sigma=sg,
                             limb_bias_pct=_summ(la, 12)["bias_pct"], apical_bias_pct=_summ(ap, 30)["bias_pct"],
                             limb_sd=_summ(la, 12)["sd"], limb_found=_summ(la, 12)["n_ok"] / reps))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- S4
GIANT_UM = 50.0


def s4_misclassification(scope, quick=False, seed=4, true_lo=45.0, true_hi=55.0):
    """P(45 um apical read as >= 50) and P(55 um read as < 50) vs sigma; and the 7/mm density call."""
    reps = 20 if quick else 80
    rng = np.random.default_rng(seed)
    rows = []
    sigmas = sorted(set([scope.sigma_um, 1, 2, 3, 4, 5, 6, 8]))
    for sg in sigmas:
        res = {}
        for tag, t in (("lo", true_lo), ("hi", true_hi)):
            m = []
            for _ in range(reps):
                img, px, cx, _roi_, _ = single_loop_image(scope, dict(limb_um=12, apical_um=t, total_um=40, limb_len=100.0), rng, sigma=sg)
                m.append(M.apical_width_fwhm(img, px, cx))
            res[tag] = np.array(m)
        p_false_giant = float((res["lo"] >= GIANT_UM).mean())
        p_missed_giant = float((res["hi"] < GIANT_UM).mean())
        lo1, hi1 = S.wilson(int((res["lo"] >= GIANT_UM).sum()), len(res["lo"]))
        lo2, hi2 = S.wilson(int((res["hi"] < GIANT_UM).sum()), len(res["hi"]))
        rows.append(dict(sigma=sg, p_45_called_giant=p_false_giant, p_55_called_not_giant=p_missed_giant,
                         p45_lo95=lo1, p45_hi95=hi1, p55_lo95=lo2, p55_hi95=hi2, n_each=len(res["lo"]),
                         err_rate=0.5 * (p_false_giant + p_missed_giant),
                         mean_45=float(res["lo"].mean()), mean_55=float(res["hi"].mean())))
    return pd.DataFrame(rows)


def s4_density_call(scope, quick=False, seed=44, texture=0.05):
    """P(true 7/mm read as < 7) and P(true 5/mm read as >= 7), at the scope's sigma.
    texture: skin texture amplitude (0.05 is a pessimistic default; measure it from real frames)."""
    reps = 10 if quick else 40
    rng = np.random.default_rng(seed)
    out = {}
    for d in (5, 7, 9):
        est = np.array([M.density_per_mm(*row_image(scope, d, rng, 1.0, texture=texture)[:2]) for _ in range(reps)])
        out[d] = est
    return dict(p_7_called_reduced=float((out[7] < 7).mean()), p_9_called_reduced=float((out[9] < 7).mean()),
                p_5_called_normal=float((out[5] >= 7).mean()), mean_est={k: float(v.mean()) for k, v in out.items()})


def max_sigma_for_error(df, limit=0.05):
    """Largest sigma such that it AND every smaller swept sigma keep the combined 45 vs 55 error rate <= limit.
    Returns dict(max_sigma_um, limit_reached): None if even the smallest fails; limit_reached False means no
    failure was found up to the top of the sweep, so the true limit is at least this value."""
    d = df.sort_values("sigma")
    good = 0.0
    last = None
    for sg, e in zip(d.sigma, d.err_rate):
        if e <= limit:
            last = float(sg)
        else:
            return dict(max_sigma_um=last, limit_reached=True)
    return dict(max_sigma_um=last, limit_reached=False)


# ----------------------------------------------------------------------------- S5
def slide_profile_sim(scope, pitch_um=10.0, tick_um=2.0, n_ticks=9, seed=5, oil=False, contrast=0.6):
    """Simulated 1D profile across a ruler with dark ticks, through the scope's forward model."""
    rng = np.random.default_rng(seed)
    px = scope.um_per_px
    res = px / math.ceil(px / 0.5 - 1e-9)
    W = math.ceil((n_ticks + 2) * pitch_um / px) * px
    H = math.ceil(60 / px) * px
    mask = np.zeros((int(round(H / res)), int(round(W / res))), bool)
    xs = (np.arange(mask.shape[1]) + 0.5) * res
    for i in range(n_ticks):
        c = (i + 1.5) * pitch_um
        mask[:, np.abs(xs - c) <= tick_um / 2] = True
    kw = scope.forward_kwargs(oil=oil, contrast=contrast)     # a printed ruler has its own contrast, not the loop's
    img = O.forward(mask, res, rng=rng, **kw)
    return img, img[img.shape[0] // 4: 3 * img.shape[0] // 4].mean(0)


def compare_profiles(real, sim):
    """RMSE between a real and a simulated tick profile after aligning by cross-correlation.
    Both are divided by their own bright level (flat-field). Returns dict(rmse, nrmse, lag_px)."""
    a = np.asarray(real, float)
    b = np.asarray(sim, float)
    a = a / np.percentile(a, 90)
    b = b / np.percentile(b, 90)
    n = min(len(a), len(b))
    best = (1e9, 0)
    for lag in range(-n // 3, n // 3 + 1):
        x = a[max(0, lag): n + min(0, lag)]
        y = b[max(0, -lag): n - max(0, lag)]
        m = min(len(x), len(y))
        if m < n // 2:
            continue
        r = float(np.sqrt(np.mean((x[:m] - y[:m]) ** 2)))
        if r < best[0]:
            best = (r, lag)
    depth = 1.0 - min(a.min(), b.min())
    return dict(rmse=best[0], nrmse=best[0] / max(depth, 1e-9), lag_px=best[1])


def s5_agreement(manual_df, tool_col="tool_um", r1="rater1_um", r2="rater2_um"):
    """Bland-Altman between raters and between the tool and the rater mean (guide: blind widths)."""
    rater_mean = (manual_df[r1] + manual_df[r2]) / 2
    return dict(rater_vs_rater=S.bland_altman(manual_df[r1], manual_df[r2]),
                tool_vs_raters=S.bland_altman(manual_df[tool_col], rater_mean),
                icc_raters=S.icc(manual_df[[r1, r2]].values))


def s5_sim_bias_vs_real(scope, manual_df, quick=False, seed=55, tool_col="tool_um", r1="rater1_um", r2="rater2_um"):
    """S5 gate on the model's tool error.

    Uses the rater-mean apical widths as the true values and simulates single measurements with the
    scope's settings. Checks (a) the simulated mean bias is within 95% uncertainty of the real tool-vs-rater-mean bias and (b) the simulated SD of the tool error is 0.5 to 2 times the real tool SD. The real tool
    SD is the Bland-Altman SD with the raters' own noise removed: var(tool - mean of raters) =
    var(tool) + var(rater)/2 and var(rater) = sd(rater1 - rater2)^2 / 2."""
    reps = 5 if quick else 20
    rng = np.random.default_rng(seed)
    rater_mean = (manual_df[r1] + manual_df[r2]) / 2
    truth = rater_mean.clip(lower=14).values
    errs = []
    for t in truth:
        for _ in range(reps):
            img, px, cx, _roi_, _ = single_loop_image(scope, dict(limb_um=min(12, t), apical_um=t, total_um=40, limb_len=100.0), rng)
            errs.append(M.apical_width_fwhm(img, px, cx) - t)
    errs = np.array(errs, float)
    errs = errs[~np.isnan(errs)]
    ba = S.bland_altman(manual_df[tool_col], rater_mean)
    rr = S.bland_altman(manual_df[r1], manual_df[r2])
    real_tool_var = ba["sd"] ** 2 - rr["sd"] ** 2 / 4.0
    real_tool_sd = float(np.sqrt(real_tool_var)) if real_tool_var > 0 else float("nan")
    mean_sim = float(errs.mean())
    sim_sd = float(errs.std(ddof=1))
    out = dict(sim_mean_bias_um=mean_sim, sim_sd_um=sim_sd, real=ba, real_tool_sd_um=real_tool_sd,
               passes=S.inside_limits(mean_sim, ba, extra_se=sim_sd / np.sqrt(len(errs))))
    out["sd_ratio"] = sim_sd / real_tool_sd if np.isfinite(real_tool_sd) and real_tool_sd > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------- S6
def s6_propagation(scope, s1_samples=None, quick=False, seed=6):
    """Width error at the scope's sigma pushed through the four Poiseuille assumptions."""
    reps = 15 if quick else 60
    rng = np.random.default_rng(seed)
    rows = []
    for w in ([8, 12, 20] if quick else G.LIMB_WIDTHS):
        meas = [_nanmean(M.limb_widths_fwhm(*[single_loop_image(scope, dict(limb_um=w, apical_um=w, total_um=w + 28, limb_len=100.0), rng)[i] for i in (0, 1, 3)]))
                for _ in range(reps)]
        mc = P.monte_carlo(w, meas)
        for name, v in mc.items():
            rows.append(dict(true_width=w, assumption=name, exponent=P.EXPONENT[name], **v))
    return pd.DataFrame(rows)


def s6_shear_table():
    """Baseline wall shear over the velocity and viscosity ranges for a few diameters (own calculation)."""
    rows = []
    for d in (8, 12, 16, 20):
        for v in (0.3, 0.6, 1.2):
            for mu in (1.2, 2.0, 3.0):
                rows.append(dict(d_um=d, v_mm_s=v, mu_mpas=mu, tau_pa=float(P.baseline_shear_pa(mu, v, d))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- S7
HEALTHY = dict(apical=(16.3, 2.4), limb=(10.8, 1.5))     # Aguiar 2006 mean, SD (verify against the paper)


def s7_repeatability(scope, quick=False, seed=7, n_subjects=None, n_repeats=None,
                     shift_um=3.0, focus_drift=0.25, illum_jitter=0.05):
    """Simulated re-seating: random subjects x repeated captures with positional jitter,
    focus drift (sigma multiplier) and lighting jitter. Returns dict(icc, ci, cv_pct, matrix).
    Compare with published intra-visit width ICC 0.91 and your own re-seated captures."""
    ns = n_subjects or (12 if quick else 30)
    nr = n_repeats or (3 if quick else 4)
    rng = np.random.default_rng(seed)
    true = rng.normal(*HEALTHY["apical"], ns).clip(10, None)
    mat = np.full((ns, nr), np.nan)
    mat_or = np.full((ns, nr), np.nan)
    for i, t in enumerate(true):
        for j in range(nr):
            sg = scope.sigma_um * float(np.exp(rng.normal(0, focus_drift)))
            dx, dy = rng.normal(0, shift_um, 2)
            ill = max(0.0, scope.illum_strength + rng.normal(0, illum_jitter))
            img, px, cx, _, _ = single_loop_image(scope, dict(limb_um=min(10.8, t), apical_um=max(t, 11), total_um=40, limb_len=100.0),
                                                  rng, sigma=sg, illum_strength=ill, shift_um=(float(dx), float(dy)))
            mat[i, j] = M.apical_width_fwhm(img, px, cx)            # tool does NOT know the shift
            mat_or[i, j] = M.apical_width_fwhm(img, px, cx + dx)    # oracle, for comparison only
    ok = ~np.isnan(mat).any(1)
    mat2 = mat[ok]
    ic = S.icc(mat2, "2,1")
    lo, hi = S.icc_ci(mat2, "2,1", n_boot=300 if quick else 1000)
    ok2 = ~np.isnan(mat_or).any(1)
    icc_or = S.icc(mat_or[ok2], '2,1') if ok2.sum() > 2 else float('nan')
    return dict(icc=ic, icc_oracle_position=icc_or, ci=(lo, hi), cv_pct=float(100 * np.nanmean(mat2.std(1, ddof=1) / mat2.mean(1))),
                n=int(ok.sum()), matrix=mat2)
