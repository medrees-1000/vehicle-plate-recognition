import csv
import tarfile
from pathlib import Path

import pytest

from alpr import ccpd

FIX = Path(__file__).parent / "fixtures"
GOOD1 = FIX / "025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33_16-37-15.jpg"
GOOD2 = FIX / "030-1_20-20&15_80&60-80&60_20&60_20&15_80&15-1_2_3_4_5_6_7-60-80.jpg"
OOB = FIX / "030-1_20-20&15_150&60-150&60_20&60_20&15_150&15-1_2_3_4_5_6_7-60-81.jpg"
CORRUPT = FIX / "030-1_20-20&15_80&60-80&60_20&60_20&15_80&15-1_2_3_4_5_6_7-60-82.jpg"


def test_parse_good() -> None:
    lab = ccpd.parse_name(GOOD1.name)
    assert lab.plate == "皖AY339S"
    assert lab.bbox == (10, 10, 50, 40) and lab.tilt == (95, 113)
    assert (lab.area, lab.brightness, lab.blur) == (25, 37, 15)
    assert lab.vertices[0] == (50, 40)


@pytest.mark.parametrize("name", [
    "notccpd.jpg",
    "025-95_113-10&10_50&40-50&40_10&40_10&10-0_0_22_27_27_33_16-37-15.jpg",  # 3 vertices
    "025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33-37-15.jpg",  # 6 plate chars
    "025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-99_0_22_27_27_33_16-37-15.jpg",  # province idx
    "025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33_99-37-15.jpg",  # ads idx
    "025-95_113-a&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33_16-37-15.jpg",  # non-int
])
def test_parse_bad(name: str) -> None:
    with pytest.raises(ValueError):
        ccpd.parse_name(name)


def test_check_image() -> None:
    assert ccpd.check_image(GOOD2).plate == "沪CDEFGH"
    with pytest.raises(ValueError, match="outside"):
        ccpd.check_image(OOB)
    with pytest.raises(OSError):
        ccpd.check_image(CORRUPT)
    with pytest.raises(OSError):
        ccpd.check_image(FIX / "nodir" / GOOD1.name)


def _name(i: int) -> str:
    return f"025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33_16-37-{i}.jpg"


def _names(per_subset: int, subsets: tuple[str, ...] = ccpd.SUBSETS + (ccpd.EXCLUDED,)) -> list[str]:
    return [f"CCPD2019/{s}/{_name(k * 100 + j)}" for k, s in enumerate(subsets) for j in range(per_subset)]


def test_sample_quotas_disjoint_deterministic() -> None:
    names = _names(10)
    rows, dropped, collapsed = ccpd.sample(names, train_per=5, heldout=10, seed=0)
    n = len(ccpd.SUBSETS)
    assert n == 8 and dropped == 0 and collapsed == 0 and len(rows) == n * 5 + 10
    assert sum(r[2] == "heldout" for r in rows) == 10
    assert {r[1] for r in rows} == set(ccpd.SUBSETS)  # np excluded
    assert len({r[0] for r in rows}) == len(rows)
    assert ccpd.sample(names, 5, 10, 0)[0] == rows
    assert ccpd.sample(names, 5, 10, 1)[0] != rows


def test_sample_drops_invalid_and_checks_size() -> None:
    names = _names(10) + [f"CCPD2019/ccpd_base/bad{i}.jpg" for i in range(3)]
    assert ccpd.sample(names, 5, 7)[1] == 3
    with pytest.raises(ValueError, match="need"):
        ccpd.sample(_names(4), train_per=5, heldout=7)


def test_sample_unexpected_subfolder_stops() -> None:
    with pytest.raises(ValueError, match="ccpd_foo"):
        ccpd.sample(_names(10, ccpd.SUBSETS + ("ccpd_foo",)), 5, 7)
    with pytest.raises(ValueError, match="unexpected subfolders"):  # a missing subset
        ccpd.sample(_names(10, ccpd.SUBSETS[1:]), 5, 7)


def test_list_build_roundtrip(tmp_path: Path) -> None:
    names = _names(2)
    archive = tmp_path / "a.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        for n in names:
            tf.add(GOOD1, arcname=n)
    assert sorted(ccpd.list_jpgs(archive)) == sorted(names)
    rows, _, _ = ccpd.sample(ccpd.list_jpgs(archive), train_per=1, heldout=7)
    manifest, out = tmp_path / "m.csv", tmp_path / "s.tar"
    ccpd.write_manifest(rows, manifest)
    assert ccpd.read_manifest(manifest) == rows
    ccpd.build(archive, ccpd.read_manifest(manifest), out)
    with tarfile.open(out) as tf:
        members = tf.getnames()
        packed = list(csv.DictReader(l.decode() for l in tf.extractfile("manifest.csv")))
    n = len(rows)  # 8 subsets x 1 train + 7 held-out
    assert n == 15 and len(members) == n + 1 and len(packed) == n
    assert sum(m.startswith("heldout/") for m in members) == 7
    assert not (tmp_path / "s.tar.part").exists()
    with pytest.raises(ValueError, match="not found"):
        ccpd.build(archive, [("CCPD2019/ccpd_base/nope.jpg", "ccpd_base", "train")], tmp_path / "x.tar")


def test_download_md5(tmp_path: Path) -> None:
    dest = tmp_path / "d.jpg"
    ccpd.download(GOOD1.as_uri(), dest, ccpd.file_md5(GOOD1))
    assert dest.read_bytes() == GOOD1.read_bytes()
    ccpd.download(GOOD1.as_uri(), dest, ccpd.file_md5(GOOD1))  # already verified: skip
    with pytest.raises(ValueError, match="MD5 mismatch"):
        ccpd.download(GOOD1.as_uri(), tmp_path / "e.jpg", "0" * 32)


def test_load_listing_caches(tmp_path: Path) -> None:
    names = _names(2)
    archive = tmp_path / "a.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        for n in names:
            tf.add(GOOD1, arcname=n)
    local, drive = tmp_path / "local.txt", tmp_path / "drive.txt"
    assert ccpd.load_listing(archive, [local, drive]) == names
    assert local.exists() and drive.exists() and not list(tmp_path.glob("*.part"))
    archive.unlink()  # a cache hit must not touch the archive
    local.unlink()
    assert ccpd.load_listing(archive, [local, drive]) == names
    assert local.exists()  # restored from the Drive copy
    drive.write_text("")
    with pytest.raises(ValueError, match="empty listing"):
        ccpd.load_listing(archive, [drive])


def test_priority_covers_subsets() -> None:
    assert sorted(ccpd.PRIORITY) == sorted(ccpd.SUBSETS)


def test_dedup_keeps_rarest_subfolder() -> None:
    n = _name(1)
    names = [f"CCPD2019/{s}/{n}" for s in ("ccpd_tilt", "ccpd_weather", "ccpd_challenge")] + [f"CCPD2019/ccpd_base/{_name(2)}"]
    for order in (names, names[::-1]):
        kept, removed = ccpd.dedup(order)
        assert removed == 2 and kept == sorted([f"CCPD2019/ccpd_weather/{n}", f"CCPD2019/ccpd_base/{_name(2)}"])


def test_sample_dedups_across_subsets() -> None:
    shared = [_name(9000 + j) for j in range(10)]
    names = _names(20) + [f"CCPD2019/{s}/{b}" for s in ccpd.SUBSETS for b in shared]
    rows, _, collapsed = ccpd.sample(names, train_per=5, heldout=8, seed=0)
    assert collapsed == 10 * (len(ccpd.SUBSETS) - 1)
    assert len({r[0].rsplit("/", 1)[1] for r in rows}) == len(rows)  # train and held-out share no name
    assert all(r[1] == "ccpd_weather" for r in rows if r[0].rsplit("/", 1)[1] in shared)


def test_sample_excludes_np_before_dedup() -> None:
    # np is not in PRIORITY: dedup would raise KeyError if np names reached it, and a name shared
    # with another subfolder must not be affected either
    names = _names(10) + [f"CCPD2019/{ccpd.EXCLUDED}/{_name(5)}", f"CCPD2019/{ccpd.EXCLUDED}/np-only.jpg"]
    rows, dropped, collapsed = ccpd.sample(names, train_per=5, heldout=8)
    assert (dropped, collapsed) == (0, 0) and all(ccpd.EXCLUDED not in r[0] for r in rows)
    with pytest.raises(KeyError):
        ccpd.dedup([f"CCPD2019/{ccpd.EXCLUDED}/{_name(5)}"])


def test_check_unique_message() -> None:
    rows = [("A/ccpd_db/x.jpg", "ccpd_db", "train"), ("A/ccpd_fn/x.jpg", "ccpd_fn", "heldout")]
    with pytest.raises(ValueError, match=r"x\.jpg in \['ccpd_db', 'ccpd_fn'\]"):
        ccpd._check_unique(rows)
