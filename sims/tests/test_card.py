"""Calibration card at nail zoom: the tick ladder, telling cards from capillaries, calibration zips, measured scale."""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import batch_run as B
import export_dataset as X
from capsim import calibration as C
from test_batch import _png

sys.path.insert(0, str(Path(__file__).parent))
from _synth_ruler import synth_card, synth_white_card_nailzoom   # noqa: E402


def _rgb(g):
    return np.stack([g] * 3, -1)


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files:
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 23, 0, 0, 0)), data)


def test_ladder_recovers_pitch_on_both_arms():
    rng = np.random.default_rng(0)
    for p, tilt in ((53.33, 1.7), (30.0, -2.6)):
        g = synth_card(rng, pitch=p, tilt=tilt, small=max(8, 0.42 * p), long=max(14, 0.72 * p))
        v, se, info = C.umpx_from_ladder(g, 100.0)
        assert info["arms"] == ["h", "v"]
        assert abs(info["pitch_px"] - p) / p < 5e-4
        assert abs(info["aspect"] - 1) < 1e-3
        assert abs(v - 100.0 / p) / (100.0 / p) < 5e-4


def test_ladder_blur_from_card_ticks():
    rng = np.random.default_rng(1)
    g = synth_card(rng, blur=1.3)
    lad = C.tick_ladder(g)
    b = C.tick_blur_sigma(None, pitch_px=lad["pitch_px"], band=lad["band"], rows=20)
    assert 0.9 < b["sigma_px"] < 1.8           # the printed line (2.2 px box) blurred by 1.3 px


def test_cards_are_rulers_and_capillaries_are_not():
    rng = np.random.default_rng(2)
    assert B.classify_image(_rgb(synth_card(rng)))[0] == "ruler"
    for s in range(4):
        f = B.matched_frame(np.random.default_rng(10 + s), 1.875, 1.5, 0.12, 0.15, 50000.0, None, 0.01)
        k, m = B.classify_image(f)
        assert k == "capillary" and m["card_ladder"] == ""


def test_calibration_zip_gives_the_measured_scale(tmp_path):
    rng = np.random.default_rng(3)
    inp = tmp_path / "in"
    inp.mkdir()
    cards = [("calibration/ruler-%d.png" % i, _png(_rgb(synth_card(rng, tilt=t)))) for i, t in enumerate((-0.7, 1.5, 2.4), 1)]
    whites = [("calibration/whitecard-%d.png" % i, _png(_rgb(synth_white_card_nailzoom(rng)))) for i in range(1, 4)]
    _zip(inp / "calibration.zip", cards + whites)
    frames = [(f"c/sam-Lmiddle-{c}-n.png", _png(B.matched_frame(rng, 1.875, 1.5, 0.3, 0.15, 50000.0, None, 0.0))) for c in (1, 2, 3)]
    _zip(inp / "caps.zip", frames)
    out = tmp_path / "batch"
    b = B.main(inp, out, sims="none", jobs=1)
    kinds = {i["input"]: i["kind"] for i in b["inputs"]}
    assert kinds == {"calibration.zip": "calibration", "caps.zip": "capillary"}
    sc = b["calibration"]["scale"]
    assert sc["status"] == "measured" and abs(sc["um_per_px"] - 100 / 53.33) < 2e-3
    st = json.loads((out / "caps" / "session_state.json").read_text())
    assert "measured" in st["scopes"] and "anat_central" not in st["scopes"]
    assert "calibration.zip" in b["calibration"]["noise_used"]["source"]
    ds = tmp_path / "ds"
    X.build(out, ds)
    assert X.check(ds) == []
    ss = pd.read_csv(ds / "tables" / "sessions.csv")
    assert ss.scale_status.iloc[0] == "measured"
    lo = pd.read_csv(ds / "tables" / "loops.csv").dropna(subset=["apical_px"])
    assert np.allclose(lo.apical_um, lo.apical_px * ss.um_per_px.iloc[0], rtol=1e-4)
    rp = pd.read_csv(ds / "calibration" / "ruler_photos.csv")
    assert rp.nail_zoom.all() and (rp.method == "crosshair tick ladder").all()
    tasks = [json.loads(l)["task"] for l in open(ds / "llm" / "sft_train.jsonl")]
    assert tasks.count("calibration_qc") == 3


FIX = Path(__file__).parent / "fixtures" / "card_nailzoom"


def test_real_card_photos():
    """The team's own card photos at nail zoom: every photo reads 1.8749 um/px on both arms, square pixels,
    and the white-card frames pass the relaxed blank test."""
    for p in sorted(FIX.glob("ruler-*.png")):
        rgb = B.load_rgb(p)
        assert B.classify_image(rgb)[0] == "ruler"
        v, se, info = C.umpx_from_ladder(rgb[..., 1].astype(float), 100.0)
        assert abs(v - 1.8749) < 5e-4 and info["arms"] == ["h", "v"] and abs(info["aspect"] - 1) < 1e-3
    for p in sorted(FIX.glob("whitecard-*.png")):
        k, m = B.classify_image(B.load_rgb(p))
        assert k != "ruler" and m["blank_relaxed"]
