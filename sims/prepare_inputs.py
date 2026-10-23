"""Turn a team capture collection into a clean batch_run input folder.

    python prepare_inputs.py "captrail images.zip" --out inputs_v2 --also sujay.zip

The collection holds folders named person<K>-unenhanced / person<K>-enhanced plus ruler/ and white/.
File names inside a folder start with person<J>, and J is NOT always K (the team restarted numbering), so two
different people can share a file name. Each folder is one person: every file is renamed to
f<K>p<J>-<hand><digit>-<capture>-<n|e>.png, which keeps names unique and the folder (the person) explicit.
Nested zips and anything else whose bytes duplicate a file already taken are skipped, so copies of earlier
uploads are counted once. Writes <out>/_MAPPING.csv (original path -> new name -> sha256); batch_run ignores
files starting with '_'. Deterministic: sorted walk, no timestamps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import shutil
import zipfile
from pathlib import Path

FOLDER = re.compile(r"^person(\d+)-(unenhanced|enhanced)$")
FILE = re.compile(r"^person(\d+)-([LR]?[a-z]+)-(\d+)-([neg])\.png$")
CAL = ("ruler", "white")


def _members(src):
    """(relative path, bytes) for every file in a zip or folder, sorted; macOS metadata dropped."""
    src = Path(src)
    if src.is_file() and src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            for i in sorted(z.infolist(), key=lambda i: i.filename):
                if not i.is_dir() and "__MACOSX" not in i.filename and not Path(i.filename).name.startswith("."):
                    yield i.filename, z.read(i)
    else:
        for p in sorted(q for q in src.rglob("*") if q.is_file()):
            if "__MACOSX" not in p.parts and not p.name.startswith("."):
                yield str(p.relative_to(src)), p.read_bytes()


def prepare(collection, out, also=()):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for rel, data in _members(collection):
        parts = Path(rel).parts
        h = hashlib.sha256(data).hexdigest()
        folder = next((p for p in parts[:-1] if FOLDER.match(p) or p in CAL), None)
        name = parts[-1]
        if folder is None or h in seen:
            rows.append(dict(source=rel, new="", sha256=h, note="skipped: duplicate bytes" if h in seen else "skipped: not in a person/ruler/white folder"))
            continue
        if folder in CAL:
            dest = Path(folder) / name
        else:
            k, view = FOLDER.match(folder).groups()
            m = FILE.match(name)
            if not m:
                rows.append(dict(source=rel, new="", sha256=h, note="skipped: unexpected file name"))
                continue
            j, site, cap, suffix = m.groups()
            dest = Path(f"f{k}-{view}") / f"f{k}p{j}-{site}-{cap}-{suffix}.png"
        (out / dest.parent).mkdir(parents=True, exist_ok=True)
        (out / dest).write_bytes(data)
        seen.add(h)
        rows.append(dict(source=rel, new=str(dest), sha256=h, note=""))
    for extra in also:
        shutil.copyfile(extra, out / Path(extra).name)
        rows.append(dict(source=str(Path(extra).name), new=Path(extra).name, sha256=hashlib.sha256(Path(extra).read_bytes()).hexdigest(), note="copied as is"))
    with open(out / "_MAPPING.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "new", "sha256", "note"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    kept = sum(1 for r in rows if r["new"])
    print(f"-> {out}: {kept} files kept, {len(rows) - kept} skipped (see _MAPPING.csv)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("collection", help="the team's capture zip or folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--also", nargs="*", default=[], help="other zips to include unchanged (e.g. an earlier session)")
    a = ap.parse_args()
    prepare(a.collection, a.out, a.also)
