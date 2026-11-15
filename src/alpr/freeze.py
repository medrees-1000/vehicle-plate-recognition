"""Freeze the test sets (M1b-7): one manifest per test set (name, MD5 of the image bytes as stored in the tar), written outside the repo.

Self-check: uv run pytest tests/test_freeze.py
"""
import csv
import hashlib
import io
import tarfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath


def tar_md5s(tar_path: str | Path, splits: Iterable[str], with_split: bool = False) -> dict[str, str]:
    """{name: MD5 of the bytes} for every file images/<split>/<file> of the tar with split in `splits`.
    name = '<split>/<file>' if with_split else '<file>'. Raises ValueError on a repeated name (message has no names)."""
    splits, out = set(splits), {}
    with tarfile.open(tar_path) as t:
        for m in t:
            parts = PurePosixPath(m.name).parts
            if m.isfile() and len(parts) == 3 and parts[0] == "images" and parts[1] in splits:
                name = f"{parts[1]}/{parts[2]}" if with_split else parts[2]
                if name in out:
                    raise ValueError("an image name occurs twice in the selected splits")
                out[name] = hashlib.md5(t.extractfile(m).read()).hexdigest()
    return out


def manifest_bytes(md5s: dict[str, str]) -> bytes:
    """CSV `name,md5`, sorted by name (Python string order), LF line endings, UTF-8."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["name", "md5"])
    w.writerows(sorted(md5s.items()))
    return buf.getvalue().encode("utf-8")


def verify(tar_path: str | Path, splits: Iterable[str], with_split: bool, manifest: str | Path) -> int:
    """Re-read the images from the tar and check the manifest file on disk matches them exactly. Returns the image count."""
    got = tar_md5s(tar_path, splits, with_split)
    if Path(manifest).read_bytes() != manifest_bytes(got):
        raise ValueError(f"the manifest does not match the {len(got)} images in the tar")
    return len(got)


def freeze(tar_path: str | Path, splits: Iterable[str], with_split: bool, out: str | Path, expected: int) -> tuple[int, str]:
    """Write the manifest to `out` (never overwrites a different one), verify it against a second read of the tar,
    return (image count, MD5 of the manifest file). Raises ValueError if the image count is not `expected`."""
    splits = tuple(splits)
    md5s = tar_md5s(tar_path, splits, with_split)
    if len(md5s) != expected:
        raise ValueError(f"expected {expected} images, found {len(md5s)}")
    data, out = manifest_bytes(md5s), Path(out)
    if out.exists():
        if out.read_bytes() != data:
            raise FileExistsError("a different manifest is already frozen at this path; not overwritten")
    else:
        part = out.with_name(out.name + ".part")
        part.write_bytes(data)
        part.rename(out)
    n = verify(tar_path, splits, with_split, out)
    return n, hashlib.md5(out.read_bytes()).hexdigest()
