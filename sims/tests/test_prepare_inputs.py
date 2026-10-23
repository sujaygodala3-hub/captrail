"""prepare_inputs: folder = person, shifted file prefixes stay apart, nested duplicates skipped, deterministic."""
import io
import zipfile

import prepare_inputs as P


def _collection(path):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("c/person1-unenhanced/person1-Lindex-1-n.png", b"A")
        z.writestr("c/person2-unenhanced/person1-Lindex-1-n.png", b"B")       # same name, different person
        z.writestr("c/person2-enhanced/person1-Lindex-1-e.png", b"C")
        z.writestr("c/ruler/ruler_1.png", b"R")
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as zi:
            zi.writestr("person1-unenhanced/person1-Lindex-1-n.png", b"A")
        z.writestr("c/person1-unenhanced.zip", inner.getvalue())             # nested copy
        z.writestr("__MACOSX/c/._x", b"junk")


def test_prepare_keeps_people_apart_and_is_deterministic(tmp_path):
    src = tmp_path / "coll.zip"
    _collection(src)
    a, b = tmp_path / "a", tmp_path / "b"
    P.prepare(src, a)
    P.prepare(src, b)
    files = sorted(str(p.relative_to(a)) for p in a.rglob("*.png"))
    assert files == ["f1-unenhanced/f1p1-Lindex-1-n.png", "f2-enhanced/f2p1-Lindex-1-e.png",
                     "f2-unenhanced/f2p1-Lindex-1-n.png", "ruler/ruler_1.png"]
    assert (a / "f1-unenhanced/f1p1-Lindex-1-n.png").read_bytes() == b"A"
    assert (a / "f2-unenhanced/f2p1-Lindex-1-n.png").read_bytes() == b"B"
    assert (a / "_MAPPING.csv").read_bytes() == (b / "_MAPPING.csv").read_bytes()
