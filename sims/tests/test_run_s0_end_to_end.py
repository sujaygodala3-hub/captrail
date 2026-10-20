"""Fake a whole S0 session with known truth, run run_s0.py, check the numbers come back."""
import json

import numpy as np
import pytest
from skimage import io

import run_s0
from capsim import geometry as G, optics as O
from capsim.scope import ScopeParams


def _save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), np.round(np.clip(img, 0, 1) * 255).astype(np.uint8), check_contrast=False)


def test_s0_recovers_known_scope(tmp_path):
    truth = ScopeParams(um_per_px=1.0, sigma_um=2.0, photons=255 / 0.25, read_noise=1.5 / 255,
                        illum_strength=0.15, illum_angle_deg=0.0, contrast=0.3)
    d = tmp_path / "s0"
    res = 0.5
    # slide: vertical ticks, 10 um pitch, 2 um wide
    W, H = 240, 120
    xs = (np.arange(int(W / res)) + 0.5) * res
    mask = np.zeros((int(H / res), int(W / res)), bool)
    for i in range(18):
        mask[:, np.abs(xs - (10 + 10 * i)) <= 1.0] = True
    for k in range(3):
        img = O.forward(mask, res, rng=np.random.default_rng(k), **{**truth.forward_kwargs(illum_strength=0.0), "contrast": 0.5})
        _save(d / "slide" / f"s{k}_raw.png", img)
    # flat card, 3 exposure levels x 4 frames, no lighting gradient
    blank = np.zeros((200, 200), bool)
    for lvl, ex in (("low", 0.25), ("mid", 0.5), ("high", 0.75)):
        for k in range(4):
            img = O.forward(blank, 1.0, rng=np.random.default_rng(10 * k + int(ex * 100)),
                            **{**truth.forward_kwargs(illum_strength=0.0), "exposure": ex})
            _save(d / "flat" / lvl / f"f{k}_raw.png", img)
    # gradient card
    _save(d / "gradient" / "g_raw.png", O.forward(np.zeros((300, 400), bool), 1.0, rng=np.random.default_rng(5),
                                                  **{**truth.forward_kwargs(), "photons": None, "read_noise": 0.0, "bits": 16}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(dict(pitch_um=10.0, tick_width_um=2.0, bits=8, min_levels=3)))
    out = tmp_path / "scope.json"
    run_s0.main(d, out, cfg)
    got = ScopeParams.load(out)
    assert got.um_per_px == pytest.approx(1.0, rel=0.03)
    assert got.sigma_um == pytest.approx(2.0, rel=0.25)
    assert got.photons == pytest.approx(255 / 0.25, rel=0.2)
    assert got.read_noise == pytest.approx(1.5 / 255, rel=0.5)
    assert got.illum_strength == pytest.approx(0.15, rel=0.25)
    for k in ("um_per_px", "sigma_um", "photons", "read_noise", "illum_strength"):
        assert got.measured[k]
    assert "contrast" in got.assumed()          # not measured, must stay flagged
