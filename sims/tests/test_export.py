"""Dataset export: deterministic, documented, leak-free splits, no real names, valid JSONL."""
import json
import zipfile

import numpy as np
import pandas as pd

import pytest

import batch_run as B
import export_dataset as X
from test_batch import _blank, _png


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files:
            z.writestr(zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0)), data)


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    return _batch(tmp_path_factory.mktemp("export"))


def _batch(tmp_path):
    rng = np.random.default_rng(11)
    inp = tmp_path / "in"
    inp.mkdir()
    sites = [("alice", "L", "index"), ("alice", "R", "middle"), ("bob", "L", "ring")]
    raw, enh = [], []
    for who, hand, digit in sites:
        for c in (1, 2):
            f = B.matched_frame(rng, 1.6, 1.9, 0.3, 0.15, 50000.0, None, 0.0)
            raw.append((f"u/{who}-{hand}{digit}-{c}-n.png", _png(f)))
            enh.append((f"e/{who}-{hand}{digit}-{c}-e.png", _png(np.clip(f.astype(float) * 1.1, 0, 255))))
    _zip(inp / "unenhanced.zip", raw)
    _zip(inp / "unenhanced copy.zip", raw)
    _zip(inp / "enhanced.zip", enh)
    _zip(inp / "white.zip", [(f"w/white_{i}.png", _png(_blank(rng))) for i in range(3)])
    out = tmp_path / "batch"
    B.main(inp, out, sims="none", jobs=1)
    return out


def test_export_is_deterministic_documented_and_private(tmp_path, batch):
    a, b = tmp_path / "ds_a", tmp_path / "ds_b"
    X.build(batch, a)
    X.build(batch, b)
    for f in sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file() and "_private" not in p.parts):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert X.check(a) == []
    fr = pd.read_csv(a / "tables" / "frames.csv")
    assert len(fr) == 12                                    # 3 sites x 2 captures x 2 views; the identical copy counted once
    assert set(fr.view) == {"raw", "enhanced"}
    assert fr.groupby("site_id").split.nunique().max() == 1
    assert set(fr.subject_id) == {"SUBJ-01", "SUBJ-02"}
    for p in a.rglob("*"):
        if p.is_file() and "_private" not in p.parts and p.suffix in (".csv", ".json", ".jsonl", ".md"):
            t = p.read_text().lower()
            assert "alice" not in t and "bob" not in t and "unenhanced copy" not in t, p
    ss = pd.read_csv(a / "tables" / "sessions.csv")
    assert len(ss) == 1 and ss.n_duplicate_inputs.iloc[0] == 1
    for s in ("train", "val", "test"):
        for line in open(a / "llm" / f"sft_{s}.jsonl"):
            r = json.loads(line)
            assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"] and r["split"] == s


def test_check_catches_tampering(tmp_path, batch):
    d = tmp_path / "ds"
    X.build(batch, d)
    f = d / "tables" / "loops.csv"
    f.write_text(f.read_text().replace("SUBJ-01", "SUBJ-09", 1))
    assert any("hash mismatch" in p for p in X.check(d))


def test_splits_keep_every_split_nonempty():
    split, _ = X.assign_splits([f"S-{i}" for i in range(7)])
    assert set(split.values()) == {"train", "val", "test"}


def test_vision_labels_stay_inside_frames(tmp_path, batch):
    d = tmp_path / "ds"
    X.build(batch, d)
    zips = batch.parent / "in"
    vo = tmp_path / "vision"
    X.build_vision(d, zips, vo)
    c = json.loads((vo / "annotations.json").read_text())
    assert c["images"] and c["annotations"]
    size = {i["id"]: (i["width"], i["height"]) for i in c["images"]}
    for a in c["annotations"]:
        x, y, w, h = a["bbox"]
        W, H = size[a["image_id"]]
        assert x >= 0 and y >= 0 and x + w <= W + 1e-6 and y + h <= H + 1e-6 and a["auto_label"]
