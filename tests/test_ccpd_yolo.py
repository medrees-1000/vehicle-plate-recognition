import io
import shutil
import tarfile
from pathlib import Path

from alpr import ccpd

FIX = Path(__file__).parent / "fixtures"  # 100x80 images
OK = "025-95_113-10&10_50&40-50&40_10&40_10&10_50&10-0_0_22_27_27_33_16-37-15.jpg"
OK2 = "030-1_20-20&15_80&60-80&60_20&60_20&15_80&15-1_2_3_4_5_6_7-60-80.jpg"
OUTSIDE = "030-1_20-20&15_150&60-150&60_20&60_20&15_150&15-1_2_3_4_5_6_7-60-81.jpg"  # x2=150 > width 100
CORRUPT = "030-1_20-20&15_80&60-80&60_20&60_20&15_80&15-1_2_3_4_5_6_7-60-82.jpg"
GAP = "025-95_113-10&10_50&40-80&40_10&40_10&10_80&10-0_0_22_27_27_33_16-37-16.jpg"  # copy of OK, vertices off
BAD = "not-a-ccpd-name.jpg"


def test_to_yolo():
    assert ccpd.to_yolo(ccpd.parse_name(OK), 100, 80) == "0 0.300000 0.312500 0.400000 0.375000"


def test_to_yolo_tar(tmp_path):
    src = tmp_path / "sample.tar"
    with tarfile.open(src, "w") as t:  # old split folders differ from the new split on purpose
        for name in (OK, OK2, OUTSIDE, CORRUPT):
            t.add(FIX / name, f"train/{name}")
        shutil.copy(FIX / OK, tmp_path / GAP)
        t.add(tmp_path / GAP, f"train/{GAP}")
        t.addfile(tarfile.TarInfo(f"train/{BAD}"), io.BytesIO(b""))
    split = {n: ("ccpd_base", s) for n, s in
             [(OK, "train"), (OK2, "val"), (OUTSIDE, "train"), (CORRUPT, "val"), (GAP, "train"), (BAD, "train"), ("gone.jpg", "val")]}
    kept, dropped = ccpd.to_yolo_tar(src, split, tmp_path / "out.tar")
    assert kept == {("train", "ccpd_base"): 1, ("val", "ccpd_base"): 1}
    assert {(b, r) for b, _, _, r in dropped} == {
        (OUTSIDE, "bbox_outside"), (CORRUPT, "corrupt"), (GAP, "box_gap"), (BAD, "unparseable"), ("gone.jpg", "missing")}
    with tarfile.open(tmp_path / "out.tar") as t:
        assert sorted(t.getnames()) == sorted([
            f"images/train/{OK}", f"labels/train/{OK[:-4]}.txt", f"images/val/{OK2}", f"labels/val/{OK2[:-4]}.txt", "data.yaml"])
        assert t.extractfile(f"labels/train/{OK[:-4]}.txt").read() == b"0 0.300000 0.312500 0.400000 0.375000\n"
