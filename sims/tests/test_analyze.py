import numpy as np
import pytest
from skimage import io

from capsim import analyze as A, geometry as G, optics as O

RES = 0.5


def make_row(n=7, apical=30.0, seed=0, sigma=2.0, contrast=0.35):
    rng = np.random.default_rng(seed)
    specs, W = G.loop_row(n, 1000 / n, rng, jitter=0.1, limb_gap=40, limb_len=100, width_art=12, width_ven=12, apex_bulge=apical - 12)
    img = O.forward(G.render_mask(specs, (W, 220), RES), RES, sigma_um=sigma, pixel_um=1.0, contrast=contrast,
                    photons=4000, read_noise=0.005, rng=rng)
    return img


def test_analyze_recovers_density_and_widths():
    info, loops = A.analyze_image(make_row(), 1.0, "up")
    assert info["n_apices"] == 7 and info["density_per_mm"] == pytest.approx(7.0, rel=0.02)
    assert info["qc_pass"]
    assert loops.apical_um.median() == pytest.approx(30, rel=0.1)
    assert loops.total_um.median() == pytest.approx(52, rel=0.08)          # limb gap 40 + limb width 12, outer to outer


def test_limb_widths_without_apical_bulge():
    info, loops = A.analyze_image(make_row(apical=12.0), 1.0, "up")
    assert np.nanmedian(loops[["limb_left_um", "limb_right_um"]].values) == pytest.approx(12, rel=0.1)


@pytest.mark.parametrize("apex,k", [("left", 1), ("down", 2), ("right", -1)])
def test_orientation_handled(apex, k):
    """Image saved with the apexes pointing left/down/right gives the same answer once told so."""
    base = make_row()
    rotated = np.rot90(base, k)           # apexes now point in the named direction
    info, loops = A.analyze_image(rotated, 1.0, apex)
    assert info["n_apices"] == 7
    assert loops.apical_um.median() == pytest.approx(30, rel=0.1)


def test_wrong_orientation_is_detectable():
    base = make_row()
    good, _ = A.analyze_image(base, 1.0, "up")
    flipped, _ = A.analyze_image(base, 1.0, "down")       # told the wrong way
    assert flipped["n_apices"] != good["n_apices"]


def test_auto_orientation_finds_up():
    name, conf, counts = A.auto_orientation(make_row(), 1.0)
    assert name == "up", counts


def test_bad_apex_argument():
    with pytest.raises(ValueError):
        A.analyze_image(make_row(), 1.0, "sideways")


def test_cli_runs(tmp_path):
    import analyze_captures as AC
    from capsim.scope import ScopeParams
    d = tmp_path / "caps"
    d.mkdir()
    io.imsave(str(d / "P01_f4_1_raw.png"), np.round(make_row() * 255).astype(np.uint8), check_contrast=False)
    sc = ScopeParams(um_per_px=1.0)
    sc.save(tmp_path / "s.json")
    AC.main(str(d), "up", str(tmp_path / "s.json"), str(tmp_path / "out"), False)
    import pandas as pd
    im = pd.read_csv(tmp_path / "out" / "images.csv")
    assert im.n_apices.iloc[0] == 7 and not im.um_per_px_measured.iloc[0]
    assert len(pd.read_csv(tmp_path / "out" / "loops.csv")) >= 5
