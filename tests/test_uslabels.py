import csv
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from alpr import uslabels


def jpg(w: int = 100, h: int = 100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 100, 50)).save(buf, "JPEG")
    return buf.getvalue()


def make_tar(path: Path, items: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as t:
        for name, data in items.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


@pytest.fixture
def tar(tmp_path: Path) -> Path:
    p = tmp_path / "us.tar"
    make_tar(p, {
        "images/test/b.jpg": jpg(), "labels/test/b.txt": b"0 0.5 0.5 0.2 0.1\n",  # one plate
        "images/test/a.jpg": jpg(), "labels/test/a.txt": b"0 0.3 0.3 0.2 0.2\n0 0.7 0.7 0.4 0.2\n",  # two plates
        "images/test/c.jpg": b"not an image", "labels/test/c.txt": b"0 0.5 0.5 0.2 0.2\n",
        "images/test/d.jpg": jpg(), "labels/test/d.txt": b"1 0.5 0.5 0.2 0.2\n",  # wrong class
        "images/test/e.jpg": jpg(),  # no label file
        "images/train/z.jpg": jpg(), "labels/train/z.txt": b"0 0.5 0.5 0.2 0.2\n",  # other split, ignored
    })
    return p


def test_crop_box_margin_min_and_clip() -> None:
    assert uslabels.crop_box((0.5, 0.5, 0.2, 0.1), 100, 100) == (38, 43, 62, 57)  # 20x10 px box, 2 px margin each side (10% of 10 is 1, min 2)
    assert uslabels.crop_box((0.5, 0.5, 0.5, 0.5), 100, 100) == (20, 20, 80, 80)  # 10% of 50 = 5
    assert uslabels.crop_box((0.05, 0.05, 0.1, 0.1), 100, 100)[:2] == (0, 0)  # clipped at the border
    with pytest.raises(ValueError):
        uslabels.crop_box((1.0, 0.5, 0.0, 0.1), 100, 100, margin=0, min_px=0)


def test_parse_labels() -> None:
    assert uslabels.parse_labels("0 0.5 0.5 0.2 0.1\n\n") == [(0.5, 0.5, 0.2, 0.1)]
    for bad in ("0 0.5 0.5 0.2", "1 0.5 0.5 0.2 0.1", "0 1.5 0.5 0.2 0.1", "0 a b c d", "0 0.5 0.5 0 0.1"):
        with pytest.raises(ValueError):
            uslabels.parse_labels(bad)


def test_make_crops(tar: Path, tmp_path: Path) -> None:
    out = tmp_path / "crops.zip"
    r = uslabels.make_crops(tar, out, expect_md5=None)
    assert (r["images"], r["plates"]) == (2, 3)
    assert dict(r["bad"]) == {"corrupt_image": 1, "bad_label": 1, "no_label_file": 1}
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["boxes_check.png", "crops/0001.png", "crops/0002.png", "crops/0003.png", "index.csv"]
        rows = list(csv.DictReader(io.StringIO(z.read("index.csv").decode())))
        assert [(x["crop_no"], x["image"], x["line"]) for x in rows] == [("1", "a", "0"), ("2", "a", "1"), ("3", "b", "0")]  # sorted by image, then line
        assert rows[0]["box_w"] == "20.0" and (rows[0]["crop_w"], rows[0]["crop_h"]) == ("24", "24")
        assert Image.open(io.BytesIO(z.read("crops/0003.png"))).size == (24, 14)
    assert not (tmp_path / "crops.zip.part").exists()


def test_make_crops_md5_mismatch_and_empty(tar: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="MD5"):
        uslabels.make_crops(tar, tmp_path / "o.zip", expect_md5="0" * 32)
    empty = tmp_path / "e.tar"
    make_tar(empty, {"images/test/x.jpg": jpg()})
    with pytest.raises(ValueError, match="no plates"):
        uslabels.make_crops(empty, tmp_path / "o2.zip", expect_md5=None)


def test_sheets_split_by_width_and_md5(tar: Path, tmp_path: Path) -> None:
    zp = tmp_path / "crops.zip"
    uslabels.make_crops(tar, zp, expect_md5=None)
    d = tmp_path / "crops"
    zipfile.ZipFile(zp).extractall(d)
    made = uslabels.make_sheets(d, tmp_path / "sheets", min_width=30)  # widths are 20, 40, 20
    assert made == {"read_01.png": [2], "small_01.png": [1, 3]}
    assert all((tmp_path / "sheets" / n).exists() for n in made)
    assert len(uslabels.make_sheets(d, tmp_path / "s2", batch=2)) == 2  # 3 crops, batch of 2
    (d / "crops" / "0001.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="MD5"):
        uslabels.make_sheets(d, tmp_path / "s3")


def test_leakcheck_finds_text_never_prints_it(tmp_path: Path) -> None:
    f = tmp_path / "labels.csv"
    f.write_text("crop_no,proposed_text,status,corrected_text,final_status\n1,AB-1234,readable,,readable\n2,XY 99 ZZ,unsure,xy99zz7,readable\n3,A1,readable,,readable\n")
    strings = uslabels.read_strings(f)
    assert strings == {"ab1234", "xy99zz", "xy99zz7"}  # compact, lower case; "A1" is too short
    texts = {"nb.ipynb": 'print("ab 12-34")', "clean.py": "x = 1", "log": "..XY99ZZ7.."}
    assert uslabels.find_leaks(strings, texts) == {"nb.ipynb": 1, "log": 2}
