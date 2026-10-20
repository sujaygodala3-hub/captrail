import json

import numpy as np
import pytest
from PIL import Image

from capsim import adapt, dataset_profile as DP, domain_gap as DG, geometry as G, morphology as MO, render_rgb as R


def test_class_order_matches_dataset_yaml():
    assert MO.CLASSES == ["bushy", "crossing", "hairpin", "tortuous"]
    p = MO.class_probs()
    assert p.argmax() == MO.CLASSES.index("hairpin") and p.sum() == pytest.approx(1)
    assert MO.class_probs(balance=1.0) == pytest.approx([0.25] * 4)


def test_crossing_limbs_actually_cross_and_hairpins_do_not():
    rng = np.random.default_rng(0)
    (cr,), _ = MO.sample_object("crossing", rng)
    pts, _, _ = G.centerline(cr)
    n_l = int(cr.limb_len / 0.25)
    art, ven = pts[:n_l, 0], pts[-n_l:, 0][::-1]
    assert np.any(np.sign(art - ven) != np.sign(art[0] - ven[0]))      # sides swap along the limb
    (hp,), _ = MO.sample_object("hairpin", rng)
    pts, _, _ = G.centerline(hp)
    art, ven = pts[:n_l, 0], pts[-n_l:, 0][::-1]
    assert np.all(art < ven)


def test_render_scene_labels_are_valid_and_deterministic():
    a = R.render_scene(np.random.default_rng(3))
    b = R.render_scene(np.random.default_rng(3))
    assert a["img"].shape == (480, 640, 3) and a["img"].dtype == np.uint8
    assert np.array_equal(a["img"], b["img"]) and a["boxes"] == b["boxes"]
    assert len(a["boxes"]) == a["meta"]["n_objects"] > 0
    for c, cx, cy, w, h in a["boxes"]:
        assert 0 <= c < 4 and 0 < cx < 1 and 0 < cy < 1 and 0 < w < 0.5 and 0 < h < 0.9
        assert cx - w / 2 >= -1e-6 and cx + w / 2 <= 1 + 1e-6 and cy - h / 2 >= -1e-6 and cy + h / 2 <= 1 + 1e-6


def test_box_matches_loop_pixels():
    """The darkest-in-green pixels should sit inside the labelled boxes (labels are geometric, so check they line up)."""
    look = R.Look(sigma_um=(1.5, 1.5), sigma_ramp=(1, 1), contrast=(0.7, 0.7), photons=(5e4, 5e4), read_noise=(0.002, 0.002),
                  bubbles=(0, 0), specks=(0, 0), texture=(0, 0), illum=(0, 0), tint_amp=(0, 0), sharpen=(0, 0), jpeg_q=(95, 95),
                  bottom_shift=(0, 0), n_objects=(1, 1))
    o = R.render_scene(np.random.default_rng(1), look, probs=np.array([0, 0, 1.0, 0]))
    g = o["img"][..., 1].astype(float)
    (c, cx, cy, w, h), = o["boxes"]
    x0, x1, y0, y1 = [int(v) for v in ((cx - w / 2) * 640, (cx + w / 2) * 640, (cy - h / 2) * 480, (cy + h / 2) * 480)]
    inside = g[y0:y1, x0:x1].mean()
    outside = np.r_[g[:, :max(0, x0 - 4)].ravel(), g[:, x1 + 4:].ravel()].mean()
    assert inside < outside - 3
    dark = g < outside - 20
    assert dark.sum() > 0 and dark[y0 - 3:y1 + 3, x0 - 3:x1 + 3].sum() / dark.sum() > 0.97      # nothing dark outside the box


def test_box_statistics_resemble_published_dataset():
    rng = np.random.default_rng(5)
    boxes = [b for _ in range(40) for b in R.render_scene(rng)["boxes"] if b[0] == 2]
    w, h = np.median([b[3] for b in boxes]), np.median([b[4] for b in boxes])
    assert 0.04 < w < 0.11 and 0.15 < h < 0.40           # dataset labels.jpg: width about 0.07, height about 0.25


def test_look_roundtrip_and_presets(tmp_path):
    lk = R.Look.celestron()
    lk.save(tmp_path / "l.json")
    assert R.Look.load(tmp_path / "l.json") == lk
    assert R.Look.dataset().sat < R.Look.celestron().sat


def test_domain_gap_detects_difference_and_not_identity():
    rng = np.random.default_rng(0)
    A = [R.render_scene(rng)["img"] for _ in range(30)]
    B = [R.render_scene(rng)["img"] for _ in range(30)]
    same = DG.two_sample(DG.feature_matrix(A), DG.feature_matrix(B))
    gray = [np.repeat(np.asarray(Image.fromarray(a).convert("L").resize((128, 96)).resize((640, 480)))[..., None], 3, 2) for a in A]
    diff = DG.two_sample(DG.feature_matrix(B), DG.feature_matrix(gray))
    assert same["auc"] < 0.75 and diff["auc"] > 0.95 and diff["auc"] > same["auc"] + 0.2
    assert "two-sample AUC" in DG.format_report(diff)


def test_profile_recovers_known_counts(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True); (tmp_path / "labels" / "train").mkdir(parents=True)
    rng = np.random.default_rng(2)
    tot = {k: 0 for k in MO.CLASSES}
    for i in range(6):
        o = R.render_scene(rng)
        Image.fromarray(o["img"]).save(tmp_path / "images" / "train" / f"a{i}.jpg")
        (tmp_path / "labels" / "train" / f"a{i}.txt").write_text("\n".join(R.yolo_lines(o["boxes"])) + "\n")
        for m in o["meta"]["objects"]:
            tot[m["cls"]] += 1
    prof = DP.profile_dataset(tmp_path / "images", tmp_path / "labels")
    assert prof["n_images"] == 6 and prof["image_sizes"] == {"640x480": 6}
    assert {k: v for k, v in prof["class_counts"].items()} == {k: v for k, v in tot.items() if v}
    assert DP.infer_px_um(prof) is not None
    json.dumps(prof)


def test_adapt_objective_and_apply():
    look = adapt.apply(R.Look.dataset(), {"contrast": 2.0, "bottom": 0.1, "sat": 0.5})
    base = R.Look.dataset()
    assert look.contrast[0] == pytest.approx(2 * base.contrast[0]) and look.sat == pytest.approx(0.5 * base.sat)
    assert look.bottom_shift[0] == pytest.approx(base.bottom_shift[0] + 0.1)
    f = np.random.default_rng(0).normal(size=(20, 15))
    assert adapt.objective(f, f) == pytest.approx(0.0, abs=1e-9)
    assert adapt.objective(f, f + 1.0) > 0.3
