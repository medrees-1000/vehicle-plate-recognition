import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from alpr import ccpd, uc3m

DATA = b"tiny fake zip"
MD5 = hashlib.md5(DATA).hexdigest()


def fake_download(content: bytes, calls: list):
    def dl(url: str, dest, md5: str) -> None:
        calls.append(url)
        Path(dest).write_bytes(content)
    return dl


def test_skips_when_drive_copy_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drive = tmp_path / "d.zip"
    drive.write_bytes(DATA)
    calls: list = []
    monkeypatch.setattr(ccpd, "download", fake_download(DATA, calls))
    uc3m.fetch(tmp_path / "l.zip", drive, "u", MD5)
    assert calls == []


def test_downloads_and_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(ccpd, "download", fake_download(DATA, calls))
    drive = tmp_path / "d.zip"
    uc3m.fetch(tmp_path / "l.zip", drive, "u", MD5)
    assert calls == ["u"] and drive.read_bytes() == DATA and not (tmp_path / "d.zip.part").exists()


def test_replaces_wrong_drive_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    drive = tmp_path / "d.zip"
    drive.write_bytes(b"corrupt")
    monkeypatch.setattr(ccpd, "download", fake_download(DATA, []))
    uc3m.fetch(tmp_path / "l.zip", drive, "u", MD5)
    assert drive.read_bytes() == DATA


def test_bad_copy_raises_and_leaves_no_drive_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ccpd, "download", fake_download(b"other", []))  # local file does not match MD5
    drive = tmp_path / "d.zip"
    with pytest.raises(ValueError):
        uc3m.fetch(tmp_path / "l.zip", drive, "u", MD5)
    assert not drive.exists() and not (tmp_path / "d.zip.part").exists()


# ---- conversion (synthetic data only) ----
def jpg(w: int = 100, h: int = 80, orientation: int | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {}
    if orientation:
        ex = Image.Exif()
        ex[0x0112] = orientation
        kw["exif"] = ex.tobytes()
    Image.new("RGB", (w, h)).save(buf, "JPEG", **kw)
    return buf.getvalue()


def lab(polys: list, w: int = 100, h: int = 80, ids: list | None = None) -> bytes:
    ids = ids or ["AA-12**567"] * len(polys)
    lps = [{"poly_coord": p, "characters": [{"char_id": 0, "bbox_coord": []}] * 5, "lp_id": i} for p, i in zip(polys, ids)]
    return json.dumps({"imageWidth": w, "imageHeight": h, "imagePath": "x.jpg", "lps": lps}).encode()


GOOD = [[10, 20], [50, 22], [52, 40], [11, 38]]


def test_polygon_to_box() -> None:
    reason, plates, size, rot = uc3m.check_item(lab([GOOD]), jpg())
    (cx, cy, w, h), lp_id, n = plates[0]
    assert (reason, size, rot, lp_id, n) == ("", (100, 80), False, "AA-12**567", 5)
    assert (cx, cy, w, h) == pytest.approx((0.31, 0.375, 0.42, 0.25))


def test_clipped_within_tolerance() -> None:
    poly = [[-0.3, 10], [50, 10], [50, 30], [-0.3, 30]]  # 0.3 px outside, tolerance is 0.5 px
    (cx, _, w, _), _, _ = uc3m.check_item(lab([poly]), jpg())[1][0]
    assert (cx, w) == pytest.approx((0.25, 0.5))


@pytest.mark.parametrize("poly, reason", [
    ([[-10, 10], [50, 10], [50, 30]], "polygon_outside"),
    ([[10, 10], [10, 20], [10, 30]], "degenerate_box"),
    ([[10, 10], [50, 10]], "unparseable"),  # two points
    ([[10, "a"], [50, 10], [50, 30]], "unparseable"),
])
def test_bad_polygons(poly: list, reason: str) -> None:
    assert uc3m.check_item(lab([poly]), jpg())[0] == reason


def test_drop_reasons() -> None:
    assert uc3m.check_item(lab([GOOD], w=200), jpg())[0] == "size_mismatch"
    assert uc3m.check_item(lab([]), jpg())[0] == "empty_lps"
    assert uc3m.check_item(b"{", jpg())[0] == "unparseable"
    assert uc3m.check_item(lab([GOOD]), b"not a jpg")[0] == "corrupt_jpg"
    assert uc3m.check_item(None, jpg())[0] == "missing_json"
    assert uc3m.check_item(lab([GOOD]), None)[0] == "missing_jpg"


def test_split_lp_counts_wildcards() -> None:
    assert uc3m.split_lp("AA-1*3*567") == ("AA", 7, 2)
    assert uc3m.split_lp("12*") == ("", 3, 1)


def test_exif_orientation_flagged() -> None:
    assert uc3m.check_item(lab([GOOD], w=80, h=100), jpg(100, 80, orientation=6))[3] is True
    assert uc3m.check_item(lab([GOOD]), jpg(orientation=1))[3] is False


def make_zip(path: Path, items: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for n, b in items.items():
            z.writestr(f"UC3M-LP/{n}", b)


def test_to_yolo_tar(tmp_path: Path) -> None:
    make_zip(tmp_path / "z.zip", {
        "train/a.json": lab([GOOD, [[60, 10], [90, 10], [90, 30], [60, 30]]]), "train/a.jpg": jpg(),
        "train/b.json": lab([GOOD]),  # no jpg
        "test/c.json": lab([GOOD]), "test/c.jpg": jpg(),
        "train.txt": b"a\nb\n",
    })
    out = tmp_path / "o.tar"
    kept, plates, dropped = uc3m.to_yolo_tar(tmp_path / "z.zip", out)
    assert (dict(kept), dict(plates), dropped) == ({"train": 1, "test": 1}, {"train": 2, "test": 1}, [("b", "train", "missing_jpg")])
    with tarfile.open(out) as t:
        assert sorted(t.getnames()) == ["data.yaml", "data_all.yaml", "images/test/c.jpg", "images/train/a.jpg", "labels/test/c.txt", "labels/train/a.txt"]
        assert len(t.extractfile("labels/train/a.txt").read().decode().splitlines()) == 2
        assert "images/train" in t.extractfile("data_all.yaml").read().decode()
    rows = list(csv.DictReader(open(tmp_path / "uc3m_labels.csv", newline="")))
    assert [(r["image_id"], r["plate_index"], r["n_visible_chars"], r["image_w"]) for r in rows] == [("c", "0", "5", "100"), ("a", "0", "5", "100"), ("a", "1", "5", "100")]  # sorted by split name, then id
    assert list(csv.reader(open(tmp_path / "o.dropped.csv")))[1] == ["b", "train", "missing_jpg"]


def test_exif_stops_conversion(tmp_path: Path) -> None:
    make_zip(tmp_path / "z.zip", {"train/a.json": lab([GOOD]), "train/a.jpg": jpg(), "test/c.json": lab([GOOD], w=80, h=100), "test/c.jpg": jpg(100, 80, orientation=6)})
    with pytest.raises(SystemExit, match="1 images have an EXIF"):
        uc3m.to_yolo_tar(tmp_path / "z.zip", tmp_path / "o.tar")
    assert list(tmp_path.glob("o.*")) == [] and not (tmp_path / "uc3m_labels.csv").exists()


def test_put(tmp_path: Path) -> None:
    src, drive = tmp_path / "s", tmp_path / "d"
    src.mkdir(), drive.mkdir()
    for n in ("o.tar", "uc3m_labels.csv", "o.dropped.csv"):
        (src / n).write_bytes(n.encode())
    with pytest.raises(SystemExit, match="free"):
        uc3m.put(src / "o.tar", drive, min_free_gb=1e9)
    assert list(drive.iterdir()) == []
    uc3m.put(src / "o.tar", drive, min_free_gb=0)
    assert (drive / "o.tar").read_bytes() == b"o.tar" and (drive / "uc3m_labels.csv").exists()
    uc3m.put(src / "o.tar", drive, min_free_gb=0)  # second run finds matching copies


def test_plate_pair_counts() -> None:
    c = uc3m.plate_pair_counts(["AD-12*45*7", "AD-1234567"], ["BN-1234*6*", "AD-9234*6*", "AD-123"])  # 3-char plate: other length, skipped
    assert dict(c) == {(3, 3): 1, (5, 5): 1, (3, 2): 1, (5, 4): 1}
