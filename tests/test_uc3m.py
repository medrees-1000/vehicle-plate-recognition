import hashlib
from pathlib import Path

import pytest

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
