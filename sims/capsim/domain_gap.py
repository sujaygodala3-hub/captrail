"""How different are synthetic frames from real ones? Image-level statistics, per-feature KS, and a
classifier two-sample test (AUC 0.5 = the classifier cannot tell them apart, 1.0 = trivially different).

Features are computed on frames area-resized to 320x240 so different capture sizes compare fairly.
This is a necessary check, not a sufficient one: low AUC does not prove the geometry is right.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FEATURES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "log_sharp", "noise", "saturation",
            "highlight", "dark", "red_minus_green", "local_contrast", "vert_gradient", "edge_energy"]
SIZE = (320, 240)


def load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def features(img, size=SIZE):
    import cv2
    a = cv2.resize(np.asarray(img), size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    g = a.mean(2)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5)
    noise = float(np.median(np.abs(hp - np.median(hp))) / 0.6745)
    lap = float(cv2.Laplacian(g, cv2.CV_32F).var())
    mx, mn = a.max(2), a.min(2)
    sat = float(np.mean((mx - mn) / (mx + 1e-6)))
    lp = cv2.GaussianBlur(g, (0, 0), 3.0)
    h = g.shape[0]
    sx, sy = cv2.Sobel(lp, cv2.CV_32F, 1, 0), cv2.Sobel(lp, cv2.CV_32F, 0, 1)
    v = [*a.mean((0, 1)), *a.std((0, 1)), np.log(lap + 1e-9), noise, sat, float((mn > 0.92).mean()),
         float((g < 0.2).mean()), float((a[..., 0] - a[..., 1]).mean()), float(np.percentile(lp, 95) - np.percentile(lp, 5)),
         float(g[2 * h // 3:].mean() - g[: h // 3].mean()), float(np.sqrt(sx ** 2 + sy ** 2).mean())]
    return np.array(v, float)


def feature_matrix(imgs, size=SIZE):
    return np.stack([features(i, size) for i in imgs])


def two_sample(real_feats, synth_feats, seed=0, folds=5):
    """Per-feature standardized mean difference and KS statistic, plus cross-validated logistic-regression AUC."""
    from scipy import stats as st
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    R, S_ = np.asarray(real_feats), np.asarray(synth_feats)
    rows = []
    for j, name in enumerate(FEATURES):
        pooled = np.sqrt(0.5 * (R[:, j].var() + S_[:, j].var())) + 1e-12
        rows.append(dict(feature=name, real_mean=float(R[:, j].mean()), synth_mean=float(S_[:, j].mean()),
                         smd=float((S_[:, j].mean() - R[:, j].mean()) / pooled),
                         ks=float(st.ks_2samp(R[:, j], S_[:, j]).statistic)))
    rng = np.random.default_rng(seed)
    n = min(len(R), len(S_))                                    # balance the classes
    Rb, Sb = R[rng.permutation(len(R))[:n]], S_[rng.permutation(len(S_))[:n]]
    X = np.vstack([Rb, Sb])
    y = np.r_[np.ones(n), np.zeros(n)]
    k = max(2, min(folds, n))
    aucs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)), n_real=int(len(R)), n_synth=int(len(S_)), n_used=int(n),
                features=rows)


def compare_dirs(real_dir, synth_imgs, max_real=500, seed=0):
    from .intake import IMG_EXT
    fs = sorted(p for p in Path(real_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
    rng = np.random.default_rng(seed)
    if len(fs) > max_real:
        fs = [fs[i] for i in rng.permutation(len(fs))[:max_real]]
    return two_sample(feature_matrix(load_rgb(p) for p in fs), feature_matrix(synth_imgs), seed=seed)


def format_report(res, top=6):
    rows = sorted(res["features"], key=lambda r: -abs(r["smd"]))[:top]
    out = [f"two-sample AUC {res['auc']:.2f} +/- {res['auc_sd']:.2f} (0.5 = indistinguishable); n_real {res['n_real']}, n_synth {res['n_synth']}",
           "largest feature gaps (synthetic minus real, in pooled SDs):"]
    out += [f"  {r['feature']:<16} real {r['real_mean']:.3f}  synth {r['synth_mean']:.3f}  smd {r['smd']:+.2f}  KS {r['ks']:.2f}" for r in rows]
    return "\n".join(out)
