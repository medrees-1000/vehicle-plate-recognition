"""US plate-text labels (M1b-5): crops of the US test plates, contact sheets for reading, and a check that no plate text leaks into git.

CLI:
    python -m alpr.uslabels crops TAR OUT_ZIP        (US test crops from us_yolo.tar: crops/NNNN.png, index.csv, boxes_check.png; prints counts only)
    python -m alpr.uslabels sheets CROPS_DIR OUT_DIR [--min-width W] [--batch 20]   (numbered contact sheets; numbers only, no file names)
    python -m alpr.uslabels counts CSV                  (final-status counts: confirmed, guess, unreadable by reason, not_plate, pending; no text)
    python -m alpr.uslabels leakcheck CSV [--history]    (prints file names and hit counts, never the text)
The text CSV (crop_no, proposed_text, status, corrected_text, final_status) holds plate text: keep it outside the repo.
"""
import argparse
import csv
import hashlib
import io
import random
import re
import subprocess
import tarfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageFont

from alpr import ccpd

TAR_MD5 = "1ce491ae6a7accefdb0b9156a30f2acc"  # us_yolo.tar (PLAN.md, M1b-4)
SPLIT = "test"  # samrat_v5 valid + test, unaltered
MARGIN = 0.10  # fraction of box width/height added on each side
MIN_MARGIN_PX = 2
SEED = 0
N_CHECK = 12  # images in boxes_check.png
INDEX_COLS = ["crop_no", "image", "line", "x1", "y1", "x2", "y2", "box_w", "box_h", "crop_w", "crop_h", "md5"]
BIN_EDGES = (20, 30, 40, 50, 60, 80, 100, 150)  # box width bins (px) printed by `crops`
MIN_LEAK_LEN = 5  # shorter plate strings are not searched (too many chance hits)
Box = tuple[float, float, float, float]  # cx, cy, w, h normalised


def parse_labels(text: str) -> list[Box]:
    """YOLO label lines `0 cx cy w h` -> boxes. Raises ValueError on a malformed line, a class other than 0 or values outside 0..1."""
    boxes = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"bad label line {ln!r}")
        cx, cy, w, h = (float(v) for v in parts[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"label values out of range: {ln!r}")
        boxes.append((cx, cy, w, h))
    return boxes


def crop_box(box: Box, w: int, h: int, margin: float = MARGIN, min_px: int = MIN_MARGIN_PX) -> tuple[int, int, int, int]:
    """Pixel box (x1, y1, x2, y2) of a normalised box plus `margin` of its width/height per side (at least min_px), clipped to the image."""
    cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    mx, my = max(min_px, margin * bw), max(min_px, margin * bh)
    x1, x2 = max(0, round(cx - bw / 2 - mx)), min(w, round(cx + bw / 2 + mx))
    y1, y2 = max(0, round(cy - bh / 2 - my)), min(h, round(cy + bh / 2 + my))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError("empty crop")
    return x1, y1, x2, y2


def _png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _check_sheet(images: list[tuple[Image.Image, list[Box]]], size: int = 400, cols: int = 4) -> Image.Image:
    """Tile images with their boxes drawn in red (for the author's box spot-check)."""
    rows = -(-len(images) // cols)
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for i, (im, boxes) in enumerate(images):
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        for cx, cy, bw, bh in boxes:
            d.rectangle([(cx - bw / 2) * im.width, (cy - bh / 2) * im.height, (cx + bw / 2) * im.width, (cy + bh / 2) * im.height], outline="red", width=max(1, im.width // 200))
        sheet.paste(im.resize((size, size)), (i % cols * size, i // cols * size))
    return sheet


def make_crops(tar_path: str | Path, out_zip: str | Path, expect_md5: str | None = TAR_MD5) -> dict:
    """Crop every plate of the SPLIT images in the tar into out_zip (via .part): crops/NNNN.png (numbered by sorted image name, then label line),
    index.csv (no plate text) and boxes_check.png. Images that are corrupt, rotated by EXIF, have no label file or a bad label are skipped and
    counted. Returns counts: images, plates, bad (per reason), widths, heights (box pixels, before the margin)."""
    if expect_md5 and (got := ccpd.file_md5(tar_path)) != expect_md5:
        raise ValueError(f"{tar_path}: MD5 {got} differs from {expect_md5}")
    imgs: dict[str, bytes] = {}
    labs: dict[str, str] = {}
    with tarfile.open(tar_path) as t:
        for m in t:
            p = PurePosixPath(m.name)
            if m.isfile() and p.parent.parts[-1:] == (SPLIT,) and p.parts[0] in ("images", "labels"):
                data = t.extractfile(m).read()
                if p.parts[0] == "images":
                    imgs[p.stem] = data
                else:
                    labs[p.stem] = data.decode("utf-8", "replace")
    bad: Counter = Counter()
    rows, widths, heights, per_image, shown = [], [], [], Counter(), []
    out_zip = Path(out_zip)
    part = out_zip.with_name(out_zip.name + ".part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:
        for stem in sorted(imgs):
            if stem not in labs:
                bad["no_label_file"] += 1
                continue
            try:
                boxes = parse_labels(labs[stem])
                with Image.open(io.BytesIO(imgs[stem])) as im:
                    rotated = im.getexif().get(0x0112, 1) != 1
                    im.load()
                    im = im.convert("RGB")
            except ValueError:
                bad["bad_label"] += 1
                continue
            except Exception:  # Pillow raises many types for broken files
                bad["corrupt_image"] += 1
                continue
            if rotated:
                bad["exif_rotated"] += 1
                continue
            if not boxes:
                bad["no_plate"] += 1
                continue
            shown.append((stem, im, boxes))
            for line, box in enumerate(boxes):
                x1, y1, x2, y2 = crop_box(box, *im.size)
                png = _png(im.crop((x1, y1, x2, y2)))
                n = len(rows) + 1
                z.writestr(f"crops/{n:04d}.png", png)
                widths.append(box[2] * im.width)
                heights.append(box[3] * im.height)
                rows.append([n, stem, line, x1, y1, x2, y2, f"{widths[-1]:.1f}", f"{heights[-1]:.1f}", x2 - x1, y2 - y1, hashlib.md5(png).hexdigest()])
            per_image[len(boxes)] += 1
        if not rows:
            raise ValueError(f"{tar_path}: no plates found in the {SPLIT} split")
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(INDEX_COLS)
        w.writerows(rows)
        z.writestr("index.csv", buf.getvalue())
        pick = random.Random(SEED).sample(shown, min(N_CHECK, len(shown)))
        z.writestr("boxes_check.png", _png(_check_sheet([(im, b) for _, im, b in pick])))
    part.replace(out_zip)
    return {"images": sum(per_image.values()), "plates": len(rows), "per_image": per_image, "bad": bad, "widths": widths, "heights": heights}


def describe(values: list[float]) -> str:
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return f"min {v[0]:.1f} | p10 {q(.1):.1f} | p25 {q(.25):.1f} | median {q(.5):.1f} | p75 {q(.75):.1f} | p90 {q(.9):.1f} | max {v[-1]:.1f}"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without sized default font
        return ImageFont.load_default()


def contact_sheet(crops: list[tuple[int, Image.Image]], cols: int = 4, cell: tuple[int, int] = (360, 140), label_h: int = 24) -> Image.Image:
    """Grid of upscaled crops (aspect kept); the number is printed above each crop, never over it. No file names."""
    rows = -(-len(crops) // cols)
    cw, ch = cell
    sheet = Image.new("RGB", (cols * cw, rows * (ch + label_h)), "white")
    d, font = ImageDraw.Draw(sheet), _font(18)
    for i, (n, im) in enumerate(crops):
        x, y = i % cols * cw, i // cols * (ch + label_h)
        d.text((x + 6, y + 2), str(n), fill="black", font=font)
        s = min((cw - 8) / im.width, ch / im.height)
        sheet.paste(im.convert("RGB").resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS), (x + 4, y + label_h))
        d.rectangle([x, y, x + cw - 1, y + ch + label_h - 1], outline="#bbbbbb")
    return sheet


def make_sheets(crops_dir: str | Path, out_dir: str | Path, min_width: float = 0, batch: int = 20) -> dict[str, list[int]]:
    """Numbered sheets from `crops_dir` (unzipped us_crops.zip). Plates with box_w >= min_width go to read_NN.png, the rest to small_NN.png
    (for the author to confirm the size rule). Returns {sheet file name: crop numbers}."""
    crops_dir, out_dir = Path(crops_dir), Path(out_dir)
    with open(crops_dir / "index.csv", newline="") as f:
        index = [(int(r["crop_no"]), float(r["box_w"]), r["md5"]) for r in csv.DictReader(f)]
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {"read": [r for r in index if r[1] >= min_width], "small": [r for r in index if r[1] < min_width]}
    made: dict[str, list[int]] = {}
    for name, items in groups.items():
        for k in range(0, len(items), batch):
            chunk = items[k : k + batch]
            crops = []
            for n, _, md5 in chunk:
                data = (crops_dir / "crops" / f"{n:04d}.png").read_bytes()
                if hashlib.md5(data).hexdigest() != md5:
                    raise ValueError(f"crop {n}: MD5 differs from index.csv")
                crops.append((n, Image.open(io.BytesIO(data))))
            fname = f"{name}_{k // batch + 1:02d}.png"
            contact_sheet(crops).save(out_dir / fname)
            made[fname] = [r[0] for r in chunk]
    return made


FINAL = ("readable", "guess", "unreadable", "not_plate")  # final_status values; empty = not reviewed yet. guess = too blurry to read, proposed text kept


def counts(csv_path: str | Path) -> dict:
    """Final-status groups of the text CSV: confirmed (readable), guess, unreadable (per reason), not_plate, pending (crop numbers).
    Scoring: the main number uses confirmed only; a second number may add guess. Raises ValueError on an unknown final_status."""
    out: dict = {"confirmed": 0, "guess": 0, "unreadable": Counter(), "not_plate": 0, "pending": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fin = r["final_status"]
            if fin not in ("", *FINAL):
                raise ValueError(f"crop {r['crop_no']}: unknown final_status {fin!r}")
            if not fin:
                out["pending"].append(int(r["crop_no"]))
            elif fin == "unreadable":
                out["unreadable"][r["reason"]] += 1
            else:
                out["confirmed" if fin == "readable" else fin] += 1
    return out


def read_strings(csv_path: str | Path) -> set[str]:
    """Plate strings (proposed and corrected) from the text CSV that are searched for, in compact form (see _compact)."""
    out = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for col in ("proposed_text", "corrected_text"):
                s = _compact(r.get(col) or "")
                if len(s) >= MIN_LEAK_LEN:
                    out.add(s)
    return out


def _compact(s: str) -> str:
    """The match rule of the OCR evaluation: case-insensitive, ignoring spaces, hyphens, dots, colons and middle dots, accents stripped."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[\s\-.:\u00b7\u2022\u2219]+", "", s).lower()


def find_leaks(strings: Iterable[str], texts: dict[str, str]) -> dict[str, int]:
    """{name: number of plate strings found} for every text (file, diff, history) that contains any compact plate string."""
    strings = list(strings)
    hits = {}
    for name, text in texts.items():
        c = _compact(text)
        n = sum(s in c for s in strings)
        if n:
            hits[name] = n
    return hits


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout


def leakcheck(csv_path: str | Path, history: bool = False) -> dict[str, int]:
    """Search every tracked and untracked (not ignored) file, the staged diff and optionally all history for the plate strings."""
    strings = read_strings(csv_path)
    texts = {}
    for name in _git("ls-files", "-co", "--exclude-standard").splitlines():
        p = Path(name)
        if p.is_file():
            texts[name] = p.read_text(encoding="utf-8", errors="ignore")
    texts["(staged diff)"] = _git("diff", "--cached")
    if history:
        texts["(git log -p --all)"] = _git("log", "-p", "--all")
    print(f"searched {len(texts)} texts for {len(strings)} plate strings (length >= {MIN_LEAK_LEN})")
    return find_leaks(strings, texts)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alpr.uslabels")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops")
    c.add_argument("tar")
    c.add_argument("out_zip")
    s = sub.add_parser("sheets")
    s.add_argument("crops_dir")
    s.add_argument("out_dir")
    s.add_argument("--min-width", type=float, default=0)
    s.add_argument("--batch", type=int, default=20)
    sub.add_parser("counts").add_argument("csv")
    k = sub.add_parser("leakcheck")
    k.add_argument("csv")
    k.add_argument("--history", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "crops":
        r = make_crops(a.tar, a.out_zip)
        print(f"images with plates: {r['images']} | plates: {r['plates']} | skipped images: {dict(r['bad'])}")
        print("images by plate count:", dict(sorted(r["per_image"].items())))
        print("box width px :", describe(r["widths"]))
        print("box height px:", describe(r["heights"]))
        edges = (0, *BIN_EDGES, float("inf"))
        print("box width bins:", {f"{lo}-{hi}": sum(lo <= v < hi for v in r["widths"]) for lo, hi in zip(edges, edges[1:])})
        print("plates under width:", {t: sum(v < t for v in r["widths"]) for t in (20, 30, 40)})
        print(f"{a.out_zip} MD5 {ccpd.file_md5(a.out_zip)}")
    elif a.cmd == "sheets":
        for name, nums in make_sheets(a.crops_dir, a.out_dir, a.min_width, a.batch).items():
            print(f"{name}: crops {nums[0]}-{nums[-1]} ({len(nums)})")
    elif a.cmd == "counts":
        c = counts(a.csv)
        print(f"confirmed {c['confirmed']} | guess {c['guess']} | unreadable {sum(c['unreadable'].values())} {dict(c['unreadable'])} | not_plate {c['not_plate']} | pending {len(c['pending'])}")
    else:
        leaks = leakcheck(a.csv, a.history)
        for name, n in sorted(leaks.items()):
            print(f"LEAK: {name}: {n} plate strings")
        print(f"leaks in {len(leaks)} texts")
        raise SystemExit(1 if leaks else 0)


if __name__ == "__main__":
    main()
