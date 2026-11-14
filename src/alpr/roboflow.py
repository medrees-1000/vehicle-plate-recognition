"""US training set from the three Roboflow exports: label conversion, copy and group bookkeeping,
a resumable pair search, and the YOLO tar. Keys are strings "set/folder/name"."""
import csv
import hashlib
import io
import math
import random
import tarfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

TOL = 0.01  # a box corner may overshoot [0, 1] by this much; it is clamped. More = bad label
DATA_YAML = "train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plate\n"  # no `path:`


def make_key(set_name: str, folder: str, name: str) -> str:
    if "/" in set_name + folder + name:
        raise ValueError(f"'/' not allowed in key parts: {set_name!r} {folder!r} {name!r}")
    return f"{set_name}/{folder}/{name}"


def split_key(key: str) -> tuple[str, str, str]:
    parts = key.split("/", 2)
    if len(parts) != 3:
        raise ValueError(f"not a set/folder/name key: {key!r}")
    return parts[0], parts[1], parts[2]


def source_of(fname: str) -> str:
    """Roboflow names copies `<original>.rf.<hash>.jpg`; the part before `.rf.` is the source photo."""
    return fname.split(".rf.")[0] if ".rf." in fname else Path(fname).stem


def plate_boxes(text: str, plate_ids: Iterable[int], n_classes: int) -> tuple[list[tuple[float, float, float, float]], str]:
    """Plate boxes (cx, cy, w, h) from one YOLO label file's text; other classes are dropped,
    polygons become their bounding rectangle. Returns (boxes, problem); problem is "" or the first
    bad-line kind, and then boxes is empty (the whole image is unusable)."""
    plate_ids = set(plate_ids)
    boxes = []
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        try:
            cls = int(f[0])
            v = [float(x) for x in f[1:]]
        except ValueError:
            return [], "not_numeric"
        if not 0 <= cls < n_classes:
            return [], "class_out_of_range"
        if cls not in plate_ids:
            continue
        if len(v) == 4:
            x1, y1, x2, y2 = v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2
        elif len(v) >= 6 and len(v) % 2 == 0:
            x1, x2, y1, y2 = min(v[0::2]), max(v[0::2]), min(v[1::2]), max(v[1::2])
        else:
            return [], "bad_field_count"
        if not all(math.isfinite(t) and -TOL <= t <= 1 + TOL for t in (x1, y1, x2, y2)):
            return [], "coord_out_of_range"
        x1, y1, x2, y2 = (min(max(t, 0.0), 1.0) for t in (x1, y1, x2, y2))
        if x2 <= x1 or y2 <= y1:
            return [], "degenerate_box"
        boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes, ""


def expand_copies(
    flagged: set[str], pool: Iterable[str], copy_sets: Iterable[str], cap: int | None = None,
    sizes: dict[tuple[str, str], int] | None = None,
) -> set[str]:
    """Pool keys (not already flagged) that share a source name with a flagged key, in sets where the
    name identifies the photo. A group larger than `cap` (images in the group; `sizes` maps (set, source)
    to a size, default: counted in `pool`) is not expanded: a huge group is a shared name stem, not one photo.
    shortcut: names only; usa_v1 copies that do not match themselves stay."""
    copy_sets, pool = set(copy_sets), list(pool)

    def src(k: str) -> tuple[str, str]:
        s, _, n = split_key(k)
        return s, source_of(n)

    sizes = sizes if sizes is not None else Counter(src(k) for k in pool)
    hit = {src(k) for k in flagged if split_key(k)[0] in copy_sets and (cap is None or sizes[src(k)] <= cap)}
    return {k for k in pool if k not in flagged and split_key(k)[0] in copy_sets and src(k) in hit}


def group_ids(pool: Iterable[str], links: Iterable[tuple[str, str]]) -> dict[str, str]:
    """key -> group id. Same set + source name = one group; each link (a, b) with both in the pool joins two groups."""
    parent = {k: k for k in pool}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first: dict[tuple[str, str], str] = {}
    for k in list(parent):
        s, _, n = split_key(k)
        union(k, first.setdefault((s, source_of(n)), k))
    for a, b in links:
        if a in parent and b in parent:
            union(a, b)
    return {k: find(k) for k in parent}


def pick_val(gid_of: dict[str, str], frac: float, seed: int) -> tuple[dict[str, str], list[str]]:
    """Draw `frac` of the groups (seeded). Returns ({group id: representative key}, other members of the drawn groups).
    The representative is a valid/test-folder image if the group has one (those are not augmented), else the first by key."""
    if not 0 < frac < 1:
        raise ValueError(f"frac must be in (0, 1): {frac}")
    groups: dict[str, list[str]] = {}
    for k, g in gid_of.items():
        groups.setdefault(g, []).append(k)
    if not groups:
        raise ValueError("empty pool")
    chosen = random.Random(seed).sample(sorted(groups), max(1, round(frac * len(groups))))
    reps = {g: min(groups[g], key=lambda k: (split_key(k)[1] == "train", k)) for g in chosen}
    return reps, sorted(k for g in chosen for k in groups[g] if k != reps[g])


def check_split(assign: dict[str, str], gid_of: dict[str, str], links: Iterable[tuple[str, str]]) -> None:
    """Raise ValueError if one group is on two sides or a link joins two sides. `assign`: key -> train/val/test."""
    sides: dict[str, set[str]] = {}
    for k, side in assign.items():
        if k in gid_of:
            sides.setdefault(gid_of[k], set()).add(side)
    bad = [g for g, s in sides.items() if len(s) > 1]
    if bad:
        raise ValueError(f"{len(bad)} groups on two sides, e.g. {bad[0]}")
    cross = [(a, b) for a, b in links if a in assign and b in assign and assign[a] != assign[b]]
    if cross:
        raise ValueError(f"{len(cross)} links join two sides, e.g. {cross[0]}")


def read_hits(path: str | Path) -> dict[tuple[str, str], int]:
    """(query, target) -> score from a hits file written by search_pairs; {} if absent. Repeated rows collapse."""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {(r["query"], r["target"]): int(r["score"]) for r in csv.DictReader(f)}


def _tag(targets: Iterable[str]) -> str:
    return hashlib.md5("\n".join(sorted(targets)).encode()).hexdigest()[:12]


def pending(queries: list[str], targets: list[str], done_csv: str | Path) -> list[str]:
    """Queries not yet finished by search_pairs for exactly this target list."""
    done: set[str] = set()
    if Path(done_csv).exists():
        with open(done_csv, newline="") as f:
            done = {r["query"] for r in csv.DictReader(f) if r["tag"] == _tag(targets)}
    return [q for q in queries if q not in done]


def search_pairs(
    queries: list[str], targets: list[str], score: Callable[[str, str], int],
    hits_csv: str | Path, done_csv: str | Path, keep: int, label: str, every: float = 30.0,
) -> None:
    """Score every query x target pair, append pairs with score >= keep to hits_csv, and log each finished query
    in done_csv, so a rerun after a disconnect skips finished queries (at most one query is redone). The log
    is tagged with a hash of the target list: a changed target list redoes every query (hits stay valid).
    Prints progress at least every `every` seconds."""
    tag = _tag(targets)
    todo = set(pending(queries, targets, done_csv))
    t0 = last = time.time()
    new_h, new_d = not Path(hits_csv).exists(), not Path(done_csv).exists()
    with open(hits_csv, "a", newline="") as fh, open(done_csv, "a", newline="") as fd:
        wh, wd = csv.writer(fh), csv.writer(fd)
        if new_h:
            wh.writerow(["query", "target", "score"])
        if new_d:
            wd.writerow(["query", "tag"])
        for i, q in enumerate(queries, 1):
            if q in todo:
                for j, t in enumerate(targets, 1):
                    s = score(q, t)
                    if s >= keep:
                        wh.writerow([q, t, s])
                    if time.time() - last >= every:
                        last = time.time()
                        print(f"  {label}: query {i}/{len(queries)}, target {j}/{len(targets)}, {last - t0:.0f} s elapsed", flush=True)
                fh.flush()
                wd.writerow([q, tag])
                fd.flush()
            if time.time() - last >= every or i == len(queries):
                last = time.time()
                print(f"  {label}: {i}/{len(queries)} queries, {last - t0:.0f} s elapsed", flush=True)


@dataclass(frozen=True)
class Item:
    key: str
    img: Path
    label: Path
    side: str  # train / val / test
    plate_ids: frozenset[int]
    n_classes: int


def build_tar(items: list[Item], out_tar: str | Path) -> tuple[Counter, list[tuple[str, str, str, str]]]:
    """Write images/<side>/, labels/<side>/ and data.yaml to out_tar (via .part), one class (plate = 0).
    Files are named <set>__<name>. Returns (kept per (set, side), dropped as (set, folder, name, reason)).
    Reasons: missing_label, unreadable_label, corrupt_image, bad_label:<kind>, no_plate."""
    kept: Counter = Counter()
    dropped: list[tuple[str, str, str, str]] = []
    seen: set[str] = set()
    part = Path(str(out_tar) + ".part")

    def add(dst: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        dst.addfile(info, io.BytesIO(data))

    with tarfile.open(part, "w") as dst:
        for it in items:
            s, folder, name = split_key(it.key)
            out = f"{s}__{name}"
            if out in seen:
                raise ValueError(f"duplicate output name {out}")
            seen.add(out)
            reason, lines = "", ""
            if not it.label.is_file():
                reason = "missing_label"
            else:
                try:
                    text = it.label.read_text()
                except (OSError, UnicodeDecodeError):
                    reason = "unreadable_label"
            if not reason:
                boxes, problem = plate_boxes(text, it.plate_ids, it.n_classes)
                if problem:
                    reason = f"bad_label:{problem}"
                elif not boxes:
                    reason = "no_plate"
                else:
                    lines = "".join("0 " + " ".join(f"{v:.6f}" for v in b) + "\n" for b in boxes)
            if not reason:
                try:
                    data = it.img.read_bytes()
                    with Image.open(io.BytesIO(data)) as im:
                        im.verify()
                except Exception:  # Pillow raises many types for broken files
                    reason = "corrupt_image"
            if reason:
                dropped.append((s, folder, name, reason))
                continue
            add(dst, f"images/{it.side}/{out}", data)
            add(dst, f"labels/{it.side}/{Path(out).stem}.txt", lines.encode())
            kept[(s, it.side)] += 1
        add(dst, "data.yaml", DATA_YAML.encode())
    part.rename(out_tar)
    return kept, dropped
