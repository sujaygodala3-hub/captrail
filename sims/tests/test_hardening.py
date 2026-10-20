import json

import numpy as np
import pandas as pd
import pytest

from capsim import gates, misspec, report, sensitivity as SE, stats as S, sweeps as W
from capsim.scope import ScopeParams


def test_wilson_and_mean_ci_known_values():
    lo, hi = S.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=0.002)
    lo, hi = S.wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=0.003) and hi == pytest.approx(0.596, abs=0.003)
    m, lo, hi, n = S.mean_ci([1.0, 2.0, 3.0, np.nan])
    assert m == 2.0 and n == 3 and lo < 2 < hi


def test_psf_kernels_share_per_axis_sd():
    for kind in misspec.KINDS:
        k = misspec.psf_kernel(kind, 3.0, 0.25)
        n = k.shape[0] // 2
        ax = np.arange(-n, n + 1) * 0.25
        sd = np.sqrt((k.sum(0) * ax ** 2).sum())
        assert sd == pytest.approx(3.0, rel=0.03), kind
        assert k.sum() == pytest.approx(1.0)


def test_misspecification_runs_and_flags_unresolvable_ruler():
    df = misspec.misspecification(ScopeParams(um_per_px=1.0, sigma_um=2.0), kinds=("disc",), sigmas=(2.0, 9.0), reps=3)
    assert len(df) == 2
    assert np.isfinite(df.sigma_gauss_fit.iloc[0])
    assert np.isnan(df.limb_bias_gap.iloc[1]) or np.isfinite(df.limb_bias_gap.iloc[1])    # no crash when the fit fails


def test_tornado_ranks_sigma_first_for_limb_bias():
    rows = SE.tornado(ScopeParams(sigma_um=4.0), "limb12_bias_pct", params=("sigma_um", "read_noise", "contrast"))
    assert rows[0]["param"] == "sigma_um"


def test_sample_scope_respects_measured_values_and_uncertainty():
    sc = ScopeParams()
    sc.mark(sigma_um=2.0, _sd=dict(sigma_um=0.3))
    rng = np.random.default_rng(0)
    draws = [SE.sample_scope(sc, rng) for _ in range(300)]
    s = np.array([d.sigma_um for d in draws])
    assert s.std() == pytest.approx(0.3, rel=0.2)
    assert all(d.um_per_px > 0 for d in draws)
    # measured value WITHOUT a stated SD is held fixed
    sc2 = ScopeParams().mark(sigma_um=2.0)
    assert {SE.sample_scope(sc2, rng).sigma_um for _ in range(5)} == {2.0}


def test_scope_uncertainty_roundtrip(tmp_path):
    sc = ScopeParams().mark(sigma_um=2.0, _sd=dict(sigma_um=0.2, um_per_px=None))
    sc.save(tmp_path / "s.json")
    assert ScopeParams.load(tmp_path / "s.json").uncertainty == {"sigma_um": 0.2}


def test_gates_statuses():
    s = dict(scope_tag="x", assumed=[], s5={})
    rows = gates.evaluate(s, {})
    assert not gates.validated(rows)
    good = dict(scope_tag="x", assumed=[], s5=dict(
        agreement=dict(icc_raters=0.9, tool_vs_raters=dict(n=12)),
        sim_vs_real=dict(passes=True, sim_mean_bias_um=0.1, real=dict(bias=0.0, sd=2.0, n=12, lo=-4, hi=4), sd_ratio=1.1),
        slide_profile=dict(nrmse=0.05)))
    assert gates.validated(gates.evaluate(good, {}))
    bad = json.loads(json.dumps(good))
    bad["s5"]["sim_vs_real"]["sd_ratio"] = 3.0
    assert not gates.validated(gates.evaluate(bad, {}))
    s0 = {"um_per_px": {"cv_pct": 3.0, "n": 10}, "photon_transfer": {"linear": False, "r2": 0.7, "g": 0.1}}
    st = {r["gate"]: r["status"] for r in gates.evaluate(dict(scope_tag="x", assumed=[], s5={}), s0)}
    assert st["S0 um/px repeatability"] == "FAIL" and st["S0 photon transfer linear"] == "FAIL"


def test_s5_sd_ratio_accounts_for_rater_noise():
    rng = np.random.default_rng(0)
    t = rng.normal(18, 4, 15).clip(15, None)
    df = pd.DataFrame(dict(rater1_um=t + rng.normal(0, 1, 15), rater2_um=t + rng.normal(0, 1, 15), tool_um=t + rng.normal(0, 1, 15)))
    out = W.s5_sim_bias_vs_real(ScopeParams(um_per_px=1.0, sigma_um=2.0), df, quick=True)
    assert out["real_tool_sd_um"] < out["real"]["sd"]            # rater noise removed
    assert np.isfinite(out["sd_ratio"])


def test_report_written(tmp_path):
    sc = ScopeParams()
    s = dict(scope_tag=sc.tag(), assumed=sc.assumed(), s1_min_limb_under_10pct=8.0, s5={},
             sens_mc_limb12_bias_pct=dict(mean=1.0, lo95=0.0, hi95=2.0))
    rows = gates.evaluate(s, {})
    report.write_report(tmp_path, s, sc, rows, report.manifest(sc, True, tmp_path))
    txt = (tmp_path / "REPORT.md").read_text()
    assert "MODEL NOT VALIDATED" in txt and "ASSUMED" in txt and "S5 forward-model validation" in txt
