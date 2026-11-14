"""UC3M-LP (Spanish plates) data plumbing. Test set only: never train on it, never share converted copies.

CLI:
    python -m alpr.uc3m download LOCAL_ZIP DRIVE_ZIP   (skips if the Drive copy already has the right MD5)
    python -m alpr.uc3m yolo ZIP OUT_TAR               (streams the zip; writes OUT_TAR + uc3m_labels.csv + <OUT>.dropped.csv)
    python -m alpr.uc3m copy OUT_TAR DRIVE_DIR         (tar and both CSVs to Drive via .part, MD5 of each copy checked)
uc3m_labels.csv holds partial plate text: Drive only, never print, commit or share it.
"""
import argparse
import csv
import io
import json
import math
import shutil
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from alpr import ccpd

URL = "https://zenodo.org/records/17152029/files/UC3M-LP.zip?download=1"
MD5 = "496e3b5e66b78d231f71d85c0e5e11a3"
SPLITS = ("train", "test")  # the authors' split = the zip's folders
OUTSIDE_TOL = 0.005  # polygon points may lie this fraction of the image side outside the image; they are clipped
MIN_FREE_GB = 10
DATA_YAML = "test: images/test\nnames:\n  0: plate\n"  # no `path:` key: Ultralytics then uses this file's folder
DATA_ALL_YAML = "test:\n  - images/train\n  - images/test\nnames:\n  0: plate\n"  # zero-shot run on all images
LABEL_COLS = ["image_id", "split", "plate_index", "lp_id", "n_visible_chars", "cx", "cy", "w", "h", "image_w", "image_h"]

Plate = tuple[tuple[float, float, float, float], str, int]  # (cx, cy, w, h normalised), lp_id as given, visible characters


def copy_verified(src: str | Path, dst: str | Path, md5: str | None = None) -> str:
    """Copy via .part, check the copy's MD5 against `md5` (default: the source's). Returns the MD5."""
    dst = Path(dst)
    md5 = md5 or ccpd.file_md5(src)
    part = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, part)
    got = ccpd.file_md5(part)
    if got != md5:
        part.unlink()
        raise ValueError(f"MD5 mismatch after copy to {dst}: got {got}, expected {md5}")
    part.replace(dst)
    return md5


def fetch(local: str | Path, drive: str | Path, url: str = URL, md5: str = MD5) -> None:
    """Skip if `drive` already matches `md5`; else download to `local` (MD5 checked, raises on mismatch),
    copy to `drive` via .part, check the copy's MD5 too. A Drive file with a wrong MD5 is replaced."""
    drive = Path(drive)
    if drive.exists():
        got = ccpd.file_md5(drive)
        if got == md5:
            print(f"{drive}: MD5 ok, skipping")
            return
        print(f"{drive}: MD5 {got} differs from {md5}, it will be replaced")
    ccpd.download(url, local, md5)
    copy_verified(local, drive, md5)
    print(f"{drive}: MD5 ok ({md5})")


def split_lp(lp_id: str) -> tuple[str, int, int]:
    """(prefix before the first '-', number of positions after it, number of '*'). No '-': prefix '' and all of it counts."""
    prefix, sep, rest = lp_id.partition("-")
    if not sep:
        prefix, rest = "", lp_id
    return prefix, len(rest), rest.count("*")


def plate_pair_counts(a: list[str], b: list[str]) -> Counter[tuple[int, int]]:
    """Over every pair (x in a, y in b) with the same text length (prefix dropped): count pairs by (k, e), k = positions
    where both plates are visible (not '*'), e = how many of those agree. e == k is a text match. Counts only, no text."""
    def text(lp: str) -> str:
        return lp.partition("-")[2] if "-" in lp else lp

    out: Counter[tuple[int, int]] = Counter()
    for n in {len(text(x)) for x in a} & {len(text(y)) for y in b}:
        ma = np.array([[ord(c) for c in text(x)] for x in a if len(text(x)) == n])
        mb = np.array([[ord(c) for c in text(y)] for y in b if len(text(y)) == n])
        for row in ma:  # one row at a time: memory stays len(mb) x n
            both = (row != 42) & (mb != 42)
            out.update(zip(both.sum(1).tolist(), (both & (mb == row)).sum(1).tolist()))
    return out


def _parse(js: bytes) -> tuple[int, int, list[tuple[list[tuple[float, float]], str, int]]]:
    """(width, height, [(polygon points, lp_id, number of characters)]). Raises ValueError/KeyError/TypeError if malformed."""
    d = json.loads(js)
    w, h, lps = int(d["imageWidth"]), int(d["imageHeight"]), d["lps"]
    if w <= 0 or h <= 0 or not isinstance(lps, list):
        raise ValueError("bad size or lps")
    out = []
    for p in lps:
        pts = [(float(x), float(y)) for x, y in p["poly_coord"]]
        if len(pts) < 3 or not all(math.isfinite(v) for pt in pts for v in pt) or not isinstance(p["lp_id"], str):
            raise ValueError("bad polygon or lp_id")
        out.append((pts, p["lp_id"], len(p["characters"])))
    return w, h, out


def check_item(js: bytes | None, jpg: bytes | None) -> tuple[str, list[Plate], tuple[int, int], bool]:
    """Check one image. Returns (drop reason or '', plates, jpg header size, EXIF orientation is not 1).
    Reasons: missing_json, missing_jpg, corrupt_jpg, unparseable, empty_lps, size_mismatch, polygon_outside, degenerate_box.
    A polygon within OUTSIDE_TOL of the border is clipped; one box per plate = bounding rectangle of its polygon."""
    if js is None:
        return "missing_json", [], (0, 0), False
    if jpg is None:
        return "missing_jpg", [], (0, 0), False
    try:
        with Image.open(io.BytesIO(jpg)) as im:
            size, rotated = im.size, im.getexif().get(0x0112, 1) != 1
            im.load()  # decode fully: a truncated file fails here
    except Exception:  # Pillow raises many types for broken files
        return "corrupt_jpg", [], (0, 0), False
    try:
        w, h, parsed = _parse(js)
    except (ValueError, KeyError, TypeError):
        return "unparseable", [], size, rotated
    if not parsed:
        return "empty_lps", [], size, rotated
    if (w, h) != size:
        return "size_mismatch", [], size, rotated
    plates: list[Plate] = []
    for pts, lp_id, n_chars in parsed:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if min(xs) < -OUTSIDE_TOL * w or max(xs) > w * (1 + OUTSIDE_TOL) or min(ys) < -OUTSIDE_TOL * h or max(ys) > h * (1 + OUTSIDE_TOL):
            return "polygon_outside", [], size, rotated
        x1, x2, y1, y2 = max(0.0, min(xs)), min(float(w), max(xs)), max(0.0, min(ys)), min(float(h), max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return "degenerate_box", [], size, rotated
        plates.append((((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h), lp_id, n_chars))
    return "", plates, size, rotated


def _add(dst: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    dst.addfile(info, io.BytesIO(data))


def to_yolo_tar(zip_path: str | Path, out_tar: str | Path) -> tuple[Counter, Counter, list[tuple[str, str, str]]]:
    """One streaming pass over the zip (nothing extracted to disk): images/<split>/, labels/<split>/, data.yaml and
    data_all.yaml into out_tar (via .part). Writes uc3m_labels.csv and <out>.dropped.csv beside it. If any image has an
    EXIF orientation other than 1, nothing is kept and SystemExit is raised with the count.
    Returns (kept images per split, kept plates per split, dropped as (image id, split, reason))."""
    out_tar = Path(out_tar)
    part = out_tar.with_name(out_tar.name + ".part")
    found: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    kept_img: Counter = Counter()
    kept_plates: Counter = Counter()
    dropped: list[tuple[str, str, str]] = []
    labels: list[list] = []
    rotated = 0
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            p = PurePosixPath(n)
            if p.parent.name in SPLITS and p.suffix in (".json", ".jpg"):
                found[(p.parent.name, p.stem)][p.suffix] = n
        if not found:
            raise ValueError(f"{zip_path}: no train/ or test/ .json or .jpg members")
        with tarfile.open(part, "w") as dst:
            for (split, stem), m in sorted(found.items()):
                js = z.read(m[".json"]) if ".json" in m else None
                jpg = z.read(m[".jpg"]) if ".jpg" in m else None
                reason, plates, (w, h), rot = check_item(js, jpg)
                rotated += rot
                if reason:
                    dropped.append((stem, split, reason))
                    continue
                _add(dst, f"images/{split}/{stem}.jpg", jpg)
                lines = "".join("0 " + " ".join(f"{v:.6f}" for v in box) + "\n" for box, _, _ in plates)
                _add(dst, f"labels/{split}/{stem}.txt", lines.encode())
                kept_img[split] += 1
                kept_plates[split] += len(plates)
                for i, (box, lp_id, n_chars) in enumerate(plates):
                    labels.append([stem, split, i, lp_id, n_chars, *(f"{v:.6f}" for v in box), w, h])
            _add(dst, "data.yaml", DATA_YAML.encode())
            _add(dst, "data_all.yaml", DATA_ALL_YAML.encode())
    if rotated:
        part.unlink()
        raise SystemExit(f"{rotated} images have an EXIF orientation other than 1: conversion stopped, nothing written")
    part.rename(out_tar)
    with open(out_tar.with_name("uc3m_labels.csv"), "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(LABEL_COLS)
        w_.writerows(labels)
    with open(out_tar.with_suffix(".dropped.csv"), "w", newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["image_id", "split", "reason"])
        w_.writerows(dropped)
    return kept_img, kept_plates, dropped


def put(out_tar: str | Path, drive_dir: str | Path, min_free_gb: float = MIN_FREE_GB) -> None:
    """Copy the tar and its two CSVs into drive_dir (stops if less than min_free_gb is free); MD5 of every copy checked."""
    out_tar, drive_dir = Path(out_tar), Path(drive_dir)
    free = shutil.disk_usage(drive_dir).free / 1e9
    if free < min_free_gb:
        raise SystemExit(f"only {free:.1f} GB free in {drive_dir}, need {min_free_gb}: stopped, nothing copied")
    for src in (out_tar, out_tar.with_name("uc3m_labels.csv"), out_tar.with_suffix(".dropped.csv")):
        dst = drive_dir / src.name
        md5 = ccpd.file_md5(src)
        if dst.exists() and ccpd.file_md5(dst) == md5:
            print(f"{dst.name}: already on Drive, MD5 ok ({md5})")
            continue
        copy_verified(src, dst, md5)
        print(f"{dst.name}: copied, MD5 of the copy ok ({md5})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alpr.uc3m")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("local")
    d.add_argument("drive")
    y = sub.add_parser("yolo")
    y.add_argument("zip")
    y.add_argument("out")
    c = sub.add_parser("copy")
    c.add_argument("out")
    c.add_argument("drive_dir")
    a = p.parse_args(argv)
    if a.cmd == "download":
        fetch(a.local, a.drive)
    elif a.cmd == "yolo":
        if Path(a.out).exists():
            print(f"{a.out} exists, skipping")
            return
        kept_img, kept_plates, dropped = to_yolo_tar(a.zip, a.out)
        print(f"images found: {sum(kept_img.values()) + len(dropped)} | kept: {sum(kept_img.values())} | dropped: {len(dropped)}")
        print("kept images per split:", dict(sorted(kept_img.items())), "| kept plates per split:", dict(sorted(kept_plates.items())))
        print("dropped per (reason, split):", dict(sorted(Counter((r[2], r[1]) for r in dropped).items())))
        print("images with EXIF orientation other than 1: 0 (any would have stopped the run)")
    else:
        put(a.out, a.drive_dir)


if __name__ == "__main__":
    main()
