"""CCPD2019 data plumbing: filename parsing, download, stratified sampling, packing.

CLI (each step skips if its output already exists):
    python -m alpr.ccpd download DEST
    python -m alpr.ccpd sample ARCHIVE MANIFEST [--cache LISTING ...]
    python -m alpr.ccpd build ARCHIVE MANIFEST OUT_TAR
    python -m alpr.ccpd split MANIFEST STATS_CSV OUT_CSV [--val 1000] [--heldout 300] [--seed 0]
    python -m alpr.ccpd yolo SAMPLE_TAR SPLIT_CSV OUT_TAR
"""
import argparse
import csv
import hashlib
import io
import random
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

URL = "https://zenodo.org/records/15647076/files/CCPD2019.tar.xz?download=1"
MD5 = "0dfbca0e6fcb7cb8ea720b0eae94c735"
SUBSETS = ("ccpd_base", "ccpd_blur", "ccpd_challenge", "ccpd_db", "ccpd_fn", "ccpd_rotate", "ccpd_tilt", "ccpd_weather")
EXCLUDED = "ccpd_np"  # no plates in these images
# CCPD2019 subfolders overlap (same image, same name). One label per image, rarest subfolder first.
PRIORITY = ("ccpd_weather", "ccpd_rotate", "ccpd_db", "ccpd_blur", "ccpd_fn", "ccpd_tilt", "ccpd_challenge", "ccpd_base")

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
    if dest.exists():
        print(f"{dest} exists ({dest.stat().st_size / 1e9:.1f} GB), checking MD5...", flush=True)
        if file_md5(dest) == md5:
            return
        print("MD5 differs, trying to resume the download", flush=True)
    pos = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"})
    with urllib.request.urlopen(req) as r:
        # 206 = server honoured the range; anything else means a full body, so start over
        resumed = getattr(r, "status", None) == 206
        done = pos if resumed else 0
        total = done + int(r.headers.get("Content-Length") or 0)
        print(f"downloading from {done / 1e9:.1f} GB of {total / 1e9:.1f} GB", flush=True)
        with open(dest, "ab" if resumed else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done // (1 << 30) != (done - len(chunk)) // (1 << 30):
                    print(f"  {done / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
    print("verifying MD5...", flush=True)
    got = file_md5(dest)
    if got != md5:
        raise ValueError(f"MD5 mismatch for {dest}: got {got}, expected {md5}")


def list_jpgs(archive: str | Path) -> list[str]:
    """Stream the archive once, returning jpg member paths (nothing is extracted)."""
    names: list[str] = []
    with tarfile.open(archive, "r|xz") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith(".jpg"):
                names.append(m.name)
                if len(names) % 50000 == 0:
                    print(f"  {len(names)} jpgs listed", flush=True)
    return names


def load_listing(archive: str | Path, caches: list[Path]) -> list[str]:
    """Member listing from the first existing cache file, else one streaming pass over the
    archive. Every missing cache is then written (atomically), before any validation."""
    names: list[str] = []
    for c in caches:
        if c.exists():
            names = c.read_text().splitlines()
            if not names:
                raise ValueError(f"empty listing cache {c}; delete it and rerun")
            print(f"listing read from {c}: {len(names)} names", flush=True)
            break
    else:
        names = list_jpgs(archive)
    for c in caches:
        if not c.exists():
            part = c.with_name(c.name + ".part")
            part.write_text("\n".join(names) + "\n")
            part.rename(c)
            print(f"listing cached to {c}", flush=True)
    return names


def dedup(names: list[str]) -> tuple[list[str], int]:
    """One member per basename, kept in the highest-priority subfolder.

    Returns (kept, sorted; number of files removed). Raises KeyError for a subfolder not in PRIORITY.
    """
    rank = {s: i for i, s in enumerate(PRIORITY)}
    best: dict[str, str] = {}
    for n in sorted(names):  # sorted, so ties resolve deterministically
        b, r = PurePosixPath(n).name, rank[PurePosixPath(n).parent.name]  # KeyError even without a collision
        if b not in best or r < rank[PurePosixPath(best[b]).parent.name]:
            best[b] = n
    return sorted(best.values()), len(names) - len(best)


def _check_unique(rows: list[Row]) -> None:
    where: dict[str, list[str]] = defaultdict(list)
    for member, subset, _ in rows:
        where[PurePosixPath(member).name].append(subset)
    dups = [f"{b} in {s}" for b, s in where.items() if len(s) > 1]
    if dups:
        raise ValueError(f"{len(dups)} duplicate basenames in sample, e.g. " + "; ".join(dups[:5]))


def sample(
    names: list[str], train_per: int = 2200, heldout: int = 300, seed: int = 0
) -> tuple[list[Row], int, int]:
    """Equal train quota per subset, held-out split evenly across subsets, disjoint.

    Basenames are deduplicated across subfolders first (see PRIORITY), so train and held-out
    never share a name. Returns (rows, unparseable names dropped, duplicate files collapsed).
    Raises if the archive's subfolders differ from SUBSETS (+ EXCLUDED) or a subset is too small.
    """
    seen = Counter(PurePosixPath(n).parent.name for n in names)
    if set(seen) - {EXCLUDED} != set(SUBSETS):
        raise ValueError(f"unexpected subfolders, expected {sorted(SUBSETS)} (+{EXCLUDED}); found {dict(seen)}")
    valid: list[str] = []
    dropped = 0
    for n in sorted(names):
        if PurePosixPath(n).parent.name == EXCLUDED:  # before dedup: EXCLUDED is not in PRIORITY
            continue
        try:
            parse_name(n)
        except ValueError:
            dropped += 1
            continue
        valid.append(n)
    valid, collapsed = dedup(valid)
    by_subset: dict[str, list[str]] = defaultdict(list)
    for n in valid:
        by_subset[PurePosixPath(n).parent.name].append(n)
    rng = random.Random(seed)
    per_heldout, extra = divmod(heldout, len(SUBSETS))
    rows: list[Row] = []
    for i, subset in enumerate(SUBSETS):
        n_held = per_heldout + (i < extra)
        pick = rng.sample(by_subset[subset], train_per + n_held) if len(by_subset[subset]) >= train_per + n_held else None
        if pick is None:
            raise ValueError(f"{subset}: {len(by_subset[subset])} valid images, need {train_per + n_held}")
        rows += [(n, subset, "heldout" if k < n_held else "train") for k, n in enumerate(pick)]
    _check_unique(rows)
    return rows, dropped, collapsed


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
                if len(done) % 2000 == 0:
                    print(f"  {len(done)}/{len(picks)} copied", flush=True)
        if missing := set(picks) - done:
            raise ValueError(f"{len(missing)} sampled images not found in archive, e.g. {sorted(missing)[0]}")
        buf = io.StringIO()
        csv.writer(buf).writerows([["member", "subset", "split"], *rows])
        data = buf.getvalue().encode()
        info = tarfile.TarInfo("manifest.csv")
        info.size = len(data)
        dst.addfile(info, io.BytesIO(data))
    part.rename(out_tar)


def has_box_gap(label: Label) -> bool:
    """True if the box and the bounding rectangle of the 4 vertices differ by >10% of the box's longer side."""
    xs, ys = [v[0] for v in label.vertices], [v[1] for v in label.vertices]
    x1, y1, x2, y2 = label.bbox
    gap = max(abs(min(xs) - x1), abs(min(ys) - y1), abs(max(xs) - x2), abs(max(ys) - y2))
    return gap > 0.1 * max(x2 - x1, y2 - y1)


def _quota(total: int) -> dict[str, int]:
    return {s: total // len(SUBSETS) + (i < total % len(SUBSETS)) for i, s in enumerate(SUBSETS)}


def split_by_plate(
    rows: list[Row], hashes: dict[str, int], heldout: int = 300, val: int = 1000,
    seed: int = 0, max_dist: int = 6, max_bundle: int = 20,
) -> list[tuple[str, str, str, str, int, bool]]:
    """Re-split into heldout/val/train so that no plate, and no near-duplicate image, is on two sides.

    Images are bundled if they share a plate or their dHash Hamming distance is <= max_dist
    (chains join). Whole bundles are assigned; heldout and val get an equal quota per subset
    label, and heldout never gets a bundle holding a box/vertices-gap image. A bundle larger
    than max_bundle goes wholly to train. `hashes` maps basename to a 64-bit dHash. Returns
    (member, subset, split, plate, bundle, box_gap) sorted by member. Raises ValueError on a
    bad name, a missing hash, or a quota that cannot be filled exactly.
    """
    rows = sorted(rows)
    n = len(rows)
    labels = [parse_name(m) for m, _, _ in rows]
    try:
        h = np.array([hashes[PurePosixPath(m).name] for m, _, _ in rows], dtype=np.uint64)
    except KeyError as e:
        raise ValueError(f"no dHash for {e}") from e
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, lab in enumerate(labels):
        parent[find(i)] = find(first.setdefault(lab.plate, i))
    for s in range(0, n, 500):
        d = np.bitwise_count(h[s:s + 500, None] ^ h[None, :])
        for a, b in zip(*np.nonzero(d <= max_dist)):
            parent[find(s + int(a))] = find(int(b))
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    bundles = sorted(groups.values())  # ordered by first member: deterministic ids
    gap = [has_box_gap(lab) for lab in labels]
    order = list(range(len(bundles)))
    random.Random(seed).shuffle(order)
    side = ["train"] * n

    def fill(name: str, total: int, allowed) -> None:
        left = _quota(total)
        for b in order:
            idx = bundles[b]
            if side[idx[0]] != "train" or not allowed(idx):
                continue
            need = Counter(rows[i][1] for i in idx)
            if all(c <= left.get(s, 0) for s, c in need.items()):
                for i in idx:
                    side[i] = name
                left = {s: left[s] - need.get(s, 0) for s in left}
        if short := {s: c for s, c in left.items() if c}:
            raise ValueError(f"cannot fill {name} quota exactly, missing per label: {short}")

    fill("heldout", heldout, lambda idx: len(idx) <= max_bundle and not any(gap[i] for i in idx))
    fill("val", val, lambda idx: len(idx) <= max_bundle)
    bundle_of = {i: b for b, idx in enumerate(bundles) for i in idx}
    return [(rows[i][0], rows[i][1], side[i], labels[i].plate, bundle_of[i], gap[i]) for i in range(n)]


def describe_bundles(out: list[tuple[str, str, str, str, int, bool]], hashes: dict[str, int], top: int = 10, max_dist: int = 6) -> list[str]:
    """One line per largest bundle: size, split, labels, distinct plates, and its links
    (same-plate pairs vs dHash-only pairs between different plates). No plate text."""
    by: dict[int, list[tuple[str, str, str, str, int, bool]]] = defaultdict(list)
    for r in out:
        by[r[4]].append(r)
    lines = []
    for b in sorted(by, key=lambda b: -len(by[b]))[:top]:
        rs = by[b]
        same = dhash_only = 0
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if rs[i][3] == rs[j][3]:
                    same += 1
                elif (hashes[PurePosixPath(rs[i][0]).name] ^ hashes[PurePosixPath(rs[j][0]).name]).bit_count() <= max_dist:
                    dhash_only += 1
        labels = dict(Counter(r[1] for r in rs))
        lines.append(f"{len(rs)} images, split {rs[0][2]}, {len({r[3] for r in rs})} plates, labels {labels}, "
                     f"same-plate pairs {same}, dHash-only pairs {dhash_only}")
    return lines


def to_yolo(label: Label, w: int, h: int) -> str:
    """One YOLO label line (class 0 = plate) from the filename box, normalised by image size."""
    x1, y1, x2, y2 = label.bbox
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


DATA_YAML = "train: images/train\nval: images/val\ntest: images/heldout\nnames:\n  0: plate\n"  # no `path:`: Ultralytics then uses this file's folder


def to_yolo_tar(
    sample_tar: str | Path, split: dict[str, tuple[str, str]], out_tar: str | Path
) -> tuple[Counter, list[tuple[str, str, str, str]]]:
    """One streaming pass over the sample tar: write images/<split>/ + labels/<split>/ + data.yaml
    to out_tar (via .part). `split` maps basename -> (subset, split). Images that fail a check are
    left out. Returns (kept per (split, subset), dropped as (basename, subset, split, reason)).
    Reasons: unparseable, corrupt, bbox_outside, box_gap, missing (in split but not in the tar).
    """
    kept: Counter = Counter()
    dropped: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    part = Path(str(out_tar) + ".part")

    def add(dst: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        dst.addfile(info, io.BytesIO(data))

    with tarfile.open(sample_tar, "r|") as src, tarfile.open(part, "w") as dst:
        for m in src:
            base = PurePosixPath(m.name).name
            if not m.isfile() or base not in split:
                continue
            subset, side = split[base]
            seen.add(base)
            data = src.extractfile(m).read()
            reason, line = "", ""
            try:
                label = parse_name(base)
            except ValueError:
                reason = "unparseable"
            else:
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        w, h = im.size
                        im.verify()
                except Exception:  # Pillow raises many types for broken files
                    reason = "corrupt"
                else:
                    x1, y1, x2, y2 = label.bbox
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                        reason = "bbox_outside"
                    elif has_box_gap(label):
                        reason = "box_gap"
                    else:
                        line = to_yolo(label, w, h)
            if reason:
                dropped.append((base, subset, side, reason))
                continue
            add(dst, f"images/{side}/{base}", data)
            add(dst, f"labels/{side}/{PurePosixPath(base).stem}.txt", (line + "\n").encode())
            kept[(side, subset)] += 1
        dropped += [(b, *split[b], "missing") for b in sorted(set(split) - seen)]
        add(dst, "data.yaml", DATA_YAML.encode())
    part.rename(out_tar)
    return kept, dropped


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
    s.add_argument("--cache", nargs="*", default=[], help="listing cache files (first existing is read, missing ones written)")
    b = sub.add_parser("build")
    b.add_argument("archive")
    b.add_argument("manifest")
    b.add_argument("out")
    sp = sub.add_parser("split")
    sp.add_argument("manifest")
    sp.add_argument("stats")
    sp.add_argument("out")
    sp.add_argument("--val", type=int, default=1000)
    sp.add_argument("--heldout", type=int, default=300)
    sp.add_argument("--seed", type=int, default=0)
    y = sub.add_parser("yolo")
    y.add_argument("sample_tar")
    y.add_argument("split_csv")
    y.add_argument("out")
    a = p.parse_args(argv)

    if a.cmd == "download":
        download(URL, a.dest, MD5)
        print(f"{a.dest}: MD5 ok ({MD5})")
    elif a.cmd == "sample":
        if Path(a.manifest).exists():
            print(f"{a.manifest} exists, skipping")
            return
        names = load_listing(a.archive, [Path(c) for c in a.cache])
        print("subfolders:", dict(Counter(PurePosixPath(n).parent.name for n in names)))
        rows, dropped, collapsed = sample(names, a.train_per, a.heldout, a.seed)
        write_manifest(rows, a.manifest)
        print(f"dropped unparseable: {dropped}; duplicate files collapsed: {collapsed}")
        print("per subset/split:", dict(Counter((r[1], r[2]) for r in rows)))
    elif a.cmd == "split":
        if Path(a.out).exists():
            print(f"{a.out} exists, skipping")
            return
        with open(a.stats, newline="") as f:
            hashes = {r["base"]: int(r["dhash"], 16) for r in csv.DictReader(f) if r["status"] == "ok"}
        out = split_by_plate(read_manifest(a.manifest), hashes, a.heldout, a.val, a.seed)
        print("10 largest bundles:", *describe_bundles(out, hashes), sep="\n  ")
        part = Path(a.out + ".part")
        with open(part, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["member", "subset", "split", "plate", "bundle", "box_gap"])
            w.writerows(out)
        part.rename(a.out)
        print("split sizes:", dict(Counter(r[2] for r in out)), "total", len(out))
        print("per subset/split:", dict(sorted(Counter((r[1], r[2]) for r in out).items())))
        print("largest bundles:", Counter(r[4] for r in out).most_common(10))
        print(f"box_gap images: {sum(r[5] for r in out)}; in heldout: {sum(r[5] for r in out if r[2] == 'heldout')}")
    elif a.cmd == "yolo":
        if Path(a.out).exists():
            print(f"{a.out} exists, skipping")
            return
        with open(a.split_csv, newline="") as f:
            split = {PurePosixPath(r["member"]).name: (r["subset"], r["split"]) for r in csv.DictReader(f)}
        kept, dropped = to_yolo_tar(a.sample_tar, split, a.out)
        with open(Path(a.out).with_suffix(".dropped.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["basename", "subset", "split", "reason"])
            w.writerows(dropped)
        print("in split file:", len(split), "| kept:", sum(kept.values()), "| dropped:", len(dropped))
        print("dropped per (reason, split):", dict(sorted(Counter((r[3], r[2]) for r in dropped).items())))
        print("dropped per (reason, subset):", dict(sorted(Counter((r[3], r[1]) for r in dropped).items())))
        per_split: Counter = Counter()
        for (side, _), c in kept.items():
            per_split[side] += c
        print("kept per split:", dict(sorted(per_split.items())))
        print("kept per (split, subset):", dict(sorted(kept.items())))
        print(f"{a.out}: MD5 {file_md5(a.out)}")
    else:
        if Path(a.out).exists():
            print(f"{a.out} exists, skipping")
            return
        build(a.archive, read_manifest(a.manifest), a.out)
        print(f"{a.out}: MD5 {file_md5(a.out)}")


if __name__ == "__main__":
    main()
