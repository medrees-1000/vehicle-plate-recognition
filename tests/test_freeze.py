import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from alpr import freeze


def make_tar(path: Path, files: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return path


FILES = {"images/train/b.jpg": b"bb", "images/train/a.jpg": b"aa", "images/test/b.jpg": b"tb", "images/val/z.jpg": b"zz", "labels/test/b.txt": b"x", "data.yaml": b"y"}


def test_freeze_writes_sorted_lf_manifest(tmp_path: Path) -> None:
    tar = make_tar(tmp_path / "t.tar", FILES)
    out = tmp_path / "m.csv"
    n, md5 = freeze.freeze(tar, ("train", "test"), True, out, 3)  # val, labels and data.yaml are not selected
    raw = out.read_bytes()
    assert n == 3 and md5 == hashlib.md5(raw).hexdigest() and b"\r" not in raw
    assert raw.decode().splitlines() == ["name,md5", f"test/b.jpg,{hashlib.md5(b'tb').hexdigest()}", f"train/a.jpg,{hashlib.md5(b'aa').hexdigest()}", f"train/b.jpg,{hashlib.md5(b'bb').hexdigest()}"]
    assert freeze.freeze(tar, ("train", "test"), True, out, 3) == (n, md5)  # same content: allowed again, file unchanged


def test_plain_names_and_count_check(tmp_path: Path) -> None:
    tar = make_tar(tmp_path / "t.tar", FILES)
    assert freeze.tar_md5s(tar, ["test"]) == {"b.jpg": hashlib.md5(b"tb").hexdigest()}
    with pytest.raises(ValueError, match="expected 5 images, found 1"):
        freeze.freeze(tar, ("test",), False, tmp_path / "m.csv", 5)
    assert not (tmp_path / "m.csv").exists()
    with pytest.raises(ValueError, match="twice"):  # train/b.jpg and test/b.jpg collide without the split in the name
        freeze.tar_md5s(tar, ("train", "test"), False)


def test_frozen_manifest_is_not_overwritten_and_tampering_is_found(tmp_path: Path) -> None:
    tar = make_tar(tmp_path / "t.tar", FILES)
    out = tmp_path / "m.csv"
    freeze.freeze(tar, ("test",), False, out, 1)
    changed = make_tar(tmp_path / "c.tar", {**FILES, "images/test/b.jpg": b"CHANGED"})
    with pytest.raises(FileExistsError):
        freeze.freeze(changed, ("test",), False, out, 1)
    with pytest.raises(ValueError, match="does not match"):
        freeze.verify(changed, ("test",), False, out)
    assert freeze.verify(tar, ("test",), False, out) == 1
