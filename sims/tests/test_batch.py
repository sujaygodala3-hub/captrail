"""Batch pipeline: noise spectrum, correlated noise, ruler blur, image typing, determinism."""
import io
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

import batch_run as B
from capsim import calibration as C, optics as O


def _png(a):
    b = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(b, "PNG")
    return b.getvalue()


def _ruler(rng, pitch=17, blur=1.5, shift=0):
    x = np.arange(640)
    prof = np.where(((x + shift) % pitch) < 6, 60.0, 200.0)
    im = ndi.gaussian_filter(np.tile(prof, (480, 1)), blur) + rng.normal(0, 2, (480, 640))
    return np.stack([im] * 3, -1)


def _blank(rng):
    im = 195 + 0.9 * rng.normal(size=(480, 640)) + 6 * ndi.gaussian_filter(rng.normal(size=(480, 640)), 4)
    return np.stack([im] * 3, -1)


def test_corr_field_unit_sd():
    f = O.corr_field((200, 300), 3.0, np.random.default_rng(0))
    assert abs(f.std() - 1) < 1e-6 and abs(f.mean()) < 0.1


def test_forward_corr_noise_only_when_asked():
    m = np.zeros((80, 80), bool)
    a = O.forward(m, 0.5, pixel_um=1.0, rng=np.random.default_rng(1))
    b = O.forward(m, 0.5, pixel_um=1.0, rng=np.random.default_rng(1), corr_noise=[[3.0, 0.01]])
    assert a.std() < 1e-6 and b.std() > 0.005


def test_noise_fit_reproduces_spectrum():
    rng = np.random.default_rng(2)
    fr = [190 + 0.9 * rng.normal(size=(240, 320)) + 1.2 * O.corr_field((240, 320), 2, rng) for _ in range(4)]
    s = C.blank_card_noise(fr)
    w = C.white_from_pixel_noise(s["pixel_hp_noise_dn"], s["mean_dn"], shape=(240, 320))
    f = C.fit_corr_noise(s["band_dn"], w["white_dn"], shape=(240, 320))
    for k in s["band_dn"]:
        assert abs(f["fitted_band_dn"][k] - s["band_dn"][k]) / s["band_dn"][k] < 0.15
    assert f["components"], "correlated part should be found"


def test_tick_blur_recovers_sigma():
    r = C.tick_blur_sigma(_ruler(np.random.default_rng(3))[..., 1])
    assert abs(r["sigma_px"] - 1.5) < 0.35
    assert abs(r["pitch_px"] - 17) < 0.5


def test_slide_pitch_robust_to_crosshair_long_ticks_and_dust():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from _synth_ruler import synth_ruler
    for seed, (p, b, t) in enumerate([(16.42, 1.2, 0.8), (12.3, 1.8, -1.5)]):
        g = synth_ruler(np.random.default_rng(seed), p, b, t)[..., 1]
        r = C.slide_pitch_px(g)
        assert abs(r["pitch_px"] - p) / p < 0.003
        bl = C.tick_blur_sigma(g)
        assert abs(bl["sigma_px"] - b) < 0.3
        assert B.classify_image(np.stack([g] * 3, -1))[0] == "ruler"


def test_classify_types():
    rng = np.random.default_rng(4)
    assert B.classify_image(_ruler(rng))[0] == "ruler"
    assert B.classify_image(_blank(rng))[0] == "blank"
    frame = B.matched_frame(rng, 1.6, 1.9, 0.12, 0.15, 50000.0, None, 0.01)
    assert B.classify_image(frame)[0] == "capillary"


def test_batch_is_deterministic_on_identical_inputs(tmp_path):
    rng = np.random.default_rng(5)
    inp = tmp_path / "in"
    inp.mkdir()
    frames = [B.matched_frame(rng, 1.6, 1.9, 0.12, 0.15, 50000.0, None, 0.01) for _ in range(3)]
    for name in ("s.zip", "s copy.zip"):
        with zipfile.ZipFile(inp / name, "w") as z:
            for i, f in enumerate(frames):    # fixed timestamps: zipfile otherwise stamps "now", and the two
                z.writestr(zipfile.ZipInfo(f"x/t-thumb-0{i + 1}n.png", (2026, 10, 21, 0, 0, 0)), _png(f))   # zips could differ
            z.writestr(zipfile.ZipInfo("__MACOSX/x/._t-thumb-01n.png", (2026, 10, 21, 0, 0, 0)), b"junk")
    with zipfile.ZipFile(inp / "white.zip", "w") as z:
        for i in range(3):
            z.writestr(f"w/white_{i}.png", _png(_blank(rng)))
    out = tmp_path / "out"
    b = B.main(inp, out, sims="none", jobs=1)
    kinds = {r["input"]: r["kind"] for r in b["inputs"]}
    assert kinds == {"s copy.zip": "capillary", "s.zip": "capillary", "white.zip": "blank"}
    assert b["determinism"] and all(d["identical"] for d in b["determinism"])
    assert (out / "BATCH_REPORT.md").exists()
