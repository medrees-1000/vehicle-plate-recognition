"""CCPD2019 data plumbing: filename parsing, download, stratified sampling, packing.

CLI (each step skips if its output already exists):
    python -m alpr.ccpd download DEST
    python -m alpr.ccpd sample ARCHIVE MANIFEST
    python -m alpr.ccpd build ARCHIVE MANIFEST OUT_TAR
"""
import argparse
import csv
import hashlib
import io
import random
import shutil
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

URL = "https://zenodo.org/records/15647076/files/CCPD2019.tar.xz?download=1"
MD5 = "0dfbca0e6fcb7cb8ea720b0eae94c735"
SUBSETS = ("ccpd_base", "ccpd_challenge", "ccpd_db", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather")
EXCLUDED = "ccpd_np"  # no plates in these images

PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]

Row = tuple[str, str, str]  # (archive member path, subset, split)


@dataclass(frozen=True)
class Label:
    area: int
    tilt: tuple[int, int]  # horizontal, vertical degrees
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    vertices: tuple[tuple[int, int], ...]  # right-bottom, left-bottom, left-top, right-top
    plate: str
    brightness: int
    blur: int


def parse_name(name: str) -> Label:
    """Parse a CCPD filename (area-tilt-bbox-vertices-plate-brightness-blur.jpg). Raises ValueError."""
    try:
        area, tilt, box, verts, plate, brightness, blur = PurePosixPath(name).stem.split("-")
        tilt_h, tilt_v = (int(v) for v in tilt.split("_"))
        (x1, y1), (x2, y2) = (tuple(int(v) for v in p.split("&")) for p in box.split("_"))
        vertices = tuple(tuple(int(v) for v in p.split("&")) for p in verts.split("_"))
        idx = [int(i) for i in plate.split("_")]
        if len(vertices) != 4 or any(len(v) != 2 for v in vertices):
            raise ValueError("need 4 vertices")
        if len(idx) != 7 or idx[0] >= len(PROVINCES) or idx[1] >= len(ALPHABETS) or max(idx[2:]) >= len(ADS):
            raise ValueError("bad plate indices")
        text = PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])
        return Label(int(area), (tilt_h, tilt_v), (x1, y1, x2, y2), vertices, text, int(brightness), int(blur))
    except ValueError as e:
        raise ValueError(f"bad CCPD name {name!r}: {e}") from e


def check_image(path: str | Path) -> Label:
    """Parse the filename and check the image opens and the box is inside it.

    Raises ValueError (bad label/box) or OSError (missing/corrupt image).
    """
    path = Path(path)
    label = parse_name(path.name)
    with Image.open(path) as im:
        w, h = im.size
        im.verify()
    x1, y1, x2, y2 = label.bbox
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"bbox {label.bbox} outside {w}x{h} image: {path.name}")
    return label


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str | Path, md5: str) -> None:
    """Download with resume (HTTP Range), then verify MD5. Skips if dest already matches."""
    dest = Path(dest)
    if dest.exists() and file_md5(dest) == md5:
        return
    pos = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"})
    with urllib.request.urlopen(req) as r:
        # 206 = server honoured the range; anything else means a full body, so start over
        with open(dest, "ab" if getattr(r, "status", None) == 206 else "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    got = file_md5(dest)
    if got != md5:
        raise ValueError(f"MD5 mismatch for {dest}: got {got}, expected {md5}")


def list_jpgs(archive: str | Path) -> list[str]:
    """Stream the archive once, returning jpg member paths (nothing is extracted)."""
    with tarfile.open(archive, "r|xz") as tf:
        return [m.name for m in tf if m.isfile() and m.name.lower().endswith(".jpg")]


def sample(names: list[str], train_per: int = 2200, heldout: int = 300, seed: int = 0) -> tuple[list[Row], int]:
    """Equal train quota per subset, held-out split evenly across subsets, disjoint.

    Returns (rows, number of unparseable names dropped). Raises if the archive's
    subfolders differ from SUBSETS (+ EXCLUDED) or a subset is too small.
    """
    seen = Counter(PurePosixPath(n).parent.name for n in names)
    if set(seen) - {EXCLUDED} != set(SUBSETS):
        raise ValueError(f"unexpected subfolders, expected {sorted(SUBSETS)} (+{EXCLUDED}); found {dict(seen)}")
    by_subset: dict[str, list[str]] = defaultdict(list)
    dropped = 0
    for n in sorted(names):
        subset = PurePosixPath(n).parent.name
        if subset == EXCLUDED:
            continue
        try:
            parse_name(n)
        except ValueError:
            dropped += 1
            continue
        by_subset[subset].append(n)
    rng = random.Random(seed)
    per_heldout, extra = divmod(heldout, len(SUBSETS))
    rows: list[Row] = []
    for i, subset in enumerate(SUBSETS):
        n_held = per_heldout + (i < extra)
        pick = rng.sample(by_subset[subset], train_per + n_held) if len(by_subset[subset]) >= train_per + n_held else None
        if pick is None:
            raise ValueError(f"{subset}: {len(by_subset[subset])} valid images, need {train_per + n_held}")
        rows += [(n, subset, "heldout" if k < n_held else "train") for k, n in enumerate(pick)]
    if len({PurePosixPath(r[0]).name for r in rows}) != len(rows):
        raise ValueError("duplicate basenames in sample")
    return rows, dropped


def write_manifest(rows: list[Row], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["member", "subset", "split"])
        w.writerows(rows)


def read_manifest(path: str | Path) -> list[Row]:
    with open(path, newline="") as f:
        return [(r["member"], r["subset"], r["split"]) for r in csv.DictReader(f)]


def build(archive: str | Path, rows: list[Row], out_tar: str | Path) -> None:
    """One streaming pass over the archive: copy sampled images into a plain .tar
    as <split>/<basename>, plus manifest.csv. Writes to .part then renames."""
    picks = {m: split for m, _, split in rows}
    part = Path(str(out_tar) + ".part")
    done: set[str] = set()
    with tarfile.open(archive, "r|xz") as src, tarfile.open(part, "w") as dst:
        for m in src:
            if m.name in picks:
                info = tarfile.TarInfo(f"{picks[m.name]}/{PurePosixPath(m.name).name}")
                info.size = m.size
                dst.addfile(info, src.extractfile(m))
                done.add(m.name)
        if missing := set(picks) - done:
            raise ValueError(f"{len(missing)} sampled images not found in archive, e.g. {sorted(missing)[0]}")
        buf = io.StringIO()
        csv.writer(buf).writerows([["member", "subset", "split"], *rows])
        data = buf.getvalue().encode()
        info = tarfile.TarInfo("manifest.csv")
        info.size = len(data)
        dst.addfile(info, io.BytesIO(data))
    part.rename(out_tar)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alpr.ccpd")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").add_argument("dest")
    s = sub.add_parser("sample")
    s.add_argument("archive")
    s.add_argument("manifest")
    s.add_argument("--train-per", type=int, default=2200)
    s.add_argument("--heldout", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("archive")
    b.add_argument("manifest")
    b.add_argument("out")
    a = p.parse_args(argv)

    if a.cmd == "download":
        download(URL, a.dest, MD5)
        print(f"{a.dest}: MD5 ok ({MD5})")
    elif a.cmd == "sample":
        if Path(a.manifest).exists():
            print(f"{a.manifest} exists, skipping")
            return
        names = list_jpgs(a.archive)
        print("subfolders:", dict(Counter(PurePosixPath(n).parent.name for n in names)))
        rows, dropped = sample(names, a.train_per, a.heldout, a.seed)
        write_manifest(rows, a.manifest)
        print(f"dropped unparseable: {dropped}")
        print("per subset/split:", dict(Counter((r[1], r[2]) for r in rows)))
    else:
        if Path(a.out).exists():
            print(f"{a.out} exists, skipping")
            return
        build(a.archive, read_manifest(a.manifest), a.out)
        print(f"{a.out}: MD5 {file_md5(a.out)}")


if __name__ == "__main__":
    main()
