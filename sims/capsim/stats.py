"""Agreement and classification statistics used by S2, S4, S5 and S7.

Bland-Altman (bias and 95% limits of agreement), ICC(2,1) and ICC(3,1) from the
two-way ANOVA mean squares (Shrout and Fleiss 1979), repeatability coefficient, ROC area
with a bootstrap interval. These are the quantities in Dinsdale 2017 and Berks 2021.
"""
from __future__ import annotations

import numpy as np


def bland_altman(a, b):
    """Bias and 95% limits of agreement of a - b (paired measurements)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    bias, sd = d.mean(), d.std(ddof=1)
    return dict(bias=float(bias), sd=float(sd), lo=float(bias - 1.96 * sd), hi=float(bias + 1.96 * sd), n=int(ok.sum()),
                mean_pair=float(((a + b) / 2)[ok].mean()))


def inside_limits(value, ba, extra_se=0.0):
    """True if a simulated bias is consistent with the REAL MEAN bias (S5 pass test).

    Compares against the 95% uncertainty of the mean bias, 1.96*sqrt(SD^2/n + extra_se^2), not the
    limits of agreement (which describe individual differences and would pass almost anything).
    `extra_se` is the standard error of the simulated bias."""
    se = np.sqrt(ba["sd"] ** 2 / max(ba["n"], 1) + extra_se ** 2)
    return bool(abs(value - ba["bias"]) <= 1.96 * se)


def repeatability(values):
    """Within-subject SD, CV and repeatability coefficient (1.96*sqrt(2)*SD) for repeat captures."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    sd = v.std(ddof=1)
    return dict(mean=float(v.mean()), sd=float(sd), cv_pct=float(100 * sd / v.mean()),
                rc=float(1.96 * np.sqrt(2) * sd), n=int(len(v)))


def _ms(x):
    x = np.asarray(x, float)
    n, k = x.shape
    gm = x.mean()
    ssr = k * ((x.mean(1) - gm) ** 2).sum()
    ssc = n * ((x.mean(0) - gm) ** 2).sum()
    sst = ((x - gm) ** 2).sum()
    sse = sst - ssr - ssc
    return n, k, ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1)), (sse + ssc) / (n * k - n)


def icc(x, kind="2,1"):
    """ICC from an (n subjects x k raters-or-repeats) matrix. kind '2,1' (absolute agreement,
    random raters) or '3,1' (consistency). Rows with NaN are dropped."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x).any(1)]
    n, k, msr, msc, mse, msw = _ms(x)
    if kind == "2,1":
        return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
    if kind == "3,1":
        return float((msr - mse) / (msr + (k - 1) * mse))
    if kind == "1,1":
        return float((msr - msw) / (msr + (k - 1) * msw))
    raise ValueError(kind)


def icc_ci(x, kind="2,1", n_boot=1000, seed=0):
    """Bootstrap (over subjects) 95% interval for the ICC."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x).any(1)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        s = x[rng.integers(0, len(x), len(x))]
        if np.ptp(s.mean(1)) == 0:
            continue
        vals.append(icc(s, kind))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def roc_auc(scores, labels):
    """Area under the ROC curve (Mann-Whitney). labels: 1 = positive. Higher score = more positive."""
    s, y = np.asarray(scores, float), np.asarray(labels).astype(bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def roc_auc_ci(scores, labels, n_boot=1000, seed=0):
    s, y = np.asarray(scores, float), np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        a = roc_auc(s[i], y[i])
        if not np.isnan(a):
            vals.append(a)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson(k, n, z=1.96):
    """Wilson score 95% interval for a proportion k/n (better than +-1.96*sqrt(p(1-p)/n) near 0 and 1)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def mean_ci(values, z=None):
    """Mean with a 95% t-interval, ignoring NaN. Returns (mean, lo, hi, n)."""
    from scipy import stats as st
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = v.mean()
    if n == 1:
        return float(m), float("nan"), float("nan"), 1
    h = st.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h), n
