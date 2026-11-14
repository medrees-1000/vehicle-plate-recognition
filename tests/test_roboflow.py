import tarfile

import pytest
from PIL import Image

from alpr import roboflow as rf

K = rf.make_key


def test_plate_boxes():
    ids = {1}
    b, p = rf.plate_boxes("1 0.5 0.5 0.2 0.1\n0 0.5 0.5 0.9 0.9\n", ids, 2)
    assert p == "" and b == [pytest.approx((0.5, 0.5, 0.2, 0.1))]
    boxes, p = rf.plate_boxes("1 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n", ids, 2)  # polygon -> rectangle
    assert p == "" and boxes == [pytest.approx((0.3, 0.4, 0.4, 0.4))]
    assert rf.plate_boxes("0 0.5 0.5 0.2 0.1\n", ids, 2) == ([], "")  # vehicle only
    b, _ = rf.plate_boxes("1 0.5 0.5 1.0 1.0\n", ids, 2)
    assert b == [pytest.approx((0.5, 0.5, 1.0, 1.0))]
    b, _ = rf.plate_boxes("1 0.5 0.5 1.01 1.0\n", ids, 2)  # overshoot inside TOL is clamped
    assert b == [pytest.approx((0.5, 0.5, 1.0, 1.0))]
    for text, kind in [
        ("1 a b c d\n", "not_numeric"), ("5 0.5 0.5 0.1 0.1\n", "class_out_of_range"),
        ("1 0.5 0.5 0.1\n", "bad_field_count"), ("1 0.5 0.5 1.5 0.1\n", "coord_out_of_range"),
        ("1 0.5 0.5 0 0.1\n", "degenerate_box"), ("1 nan 0.5 0.1 0.1\n", "coord_out_of_range"),
    ]:
        assert rf.plate_boxes(text, ids, 2) == ([], kind)


def test_keys():
    assert rf.split_key(K("a", "train", "x.rf.1.jpg")) == ("a", "train", "x.rf.1.jpg")
    with pytest.raises(ValueError):
        K("a", "train", "x/y.jpg")
    with pytest.raises(ValueError):
        rf.split_key("nokey")


def test_groups_copies_pick_check():
    pool = [K("fl", "train", f"{n}.rf.{h}.jpg") for n in "ab" for h in "123"] + [K("fl", "valid", "a.rf.9.jpg"), K("us", "train", "c.jpg"), K("us", "train", "d.jpg"), K("us", "valid", "e.jpg")]
    assert rf.expand_copies({K("fl", "train", "a.rf.1.jpg")}, pool, {"fl"}) == {K("fl", "train", "a.rf.2.jpg"), K("fl", "train", "a.rf.3.jpg"), K("fl", "valid", "a.rf.9.jpg")}
    assert rf.expand_copies({K("us", "train", "c.jpg")}, pool, {"fl"}) == set()  # usa: no name copies
    fa = {K("fl", "train", "a.rf.1.jpg")}
    assert rf.expand_copies(fa, pool, {"fl"}, cap=4) == rf.expand_copies(fa, pool, {"fl"})  # group of 4 is within cap 4
    assert rf.expand_copies(fa, pool, {"fl"}, cap=3) == set()  # group of 4 > cap 3: only the flagged image goes
    assert len(rf.expand_copies(fa, pool, {"fl"}, cap=3, sizes={("fl", "a"): 4 - 2})) == 3  # sizes override the count in `pool`
    assert rf.expand_copies({K("fl", "train", "b.rf.1.jpg")}, pool, {"fl"}, cap=3) == {K("fl", "train", "b.rf.2.jpg"), K("fl", "train", "b.rf.3.jpg")}  # other groups unaffected
    links = [(K("us", "train", "c.jpg"), K("us", "valid", "e.jpg")), (K("us", "valid", "e.jpg"), K("us", "train", "d.jpg")), ("gone/x/y", pool[0])]  # chain; link outside pool ignored
    g = rf.group_ids(pool, links)
    assert len(set(g.values())) == 3  # fl-a (4), fl-b (3), us chain (3)
    assert g[K("us", "train", "c.jpg")] == g[K("us", "train", "d.jpg")] == g[K("us", "valid", "e.jpg")]
    reps, copies = rf.pick_val(g, 0.34, seed=0)
    assert (reps, copies) == rf.pick_val(g, 0.34, seed=0) and len(reps) == 1
    members = [k for k in pool if g[k] in reps]
    assert sorted(copies + list(reps.values())) == sorted(members)
    rep = next(iter(reps.values()))
    if g[rep] == g[K("fl", "train", "a.rf.1.jpg")]:
        assert rep == K("fl", "valid", "a.rf.9.jpg")  # valid-folder copy is preferred
    assign = {k: ("val" if k == rep else "train") for k in pool if k not in copies}
    rf.check_split(assign, g, links)
    assign[copies[0]] = "train"
    with pytest.raises(ValueError):
        rf.check_split(assign, g, links)
    with pytest.raises(ValueError):  # link across sides, groups unknown to gid_of
        rf.check_split({"a/x/1": "train", "a/x/2": "val"}, {}, [("a/x/1", "a/x/2")])
    with pytest.raises(ValueError):
        rf.pick_val(g, 1.0, 0)


def test_search_pairs_resume(tmp_path):
    calls = []

    def score(q, t):
        calls.append((q, t))
        return 50 if (q, t) == ("q1", "t2") else 1

    hits, done = tmp_path / "h.csv", tmp_path / "d.csv"
    rf.search_pairs(["q1", "q2"], ["t1", "t2"], score, hits, done, keep=8, label="t")
    assert len(calls) == 4 and rf.read_hits(hits) == {("q1", "t2"): 50}
    assert rf.pending(["q1", "q2", "q3"], ["t1", "t2"], done) == ["q3"] and rf.pending(["q1"], ["t1"], done) == ["q1"]
    rf.search_pairs(["q1", "q2"], ["t1", "t2"], score, hits, done, keep=8, label="t")  # resume: nothing redone
    assert len(calls) == 4
    rf.search_pairs(["q1", "q2"], ["t1", "t2", "t3"], score, hits, done, keep=8, label="t")  # new target list: redo
    assert len(calls) == 10 and rf.read_hits(hits) == {("q1", "t2"): 50}
    assert rf.read_hits(tmp_path / "none.csv") == {}


def test_build_tar(tmp_path):
    def item(name, label, side="train", image=True):
        img = tmp_path / name
        if image:
            Image.new("RGB", (20, 10), "gray").save(img, "JPEG")
        else:
            img.write_bytes(b"not an image")
        lab = tmp_path / (name + ".txt")
        if label is not None:
            lab.write_text(label)
        return rf.Item(K("s1", "train", name), img, lab, side, frozenset({1}), 2)

    items = [
        item("ok.jpg", "1 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.9 0.9\n"),
        item("poly.jpg", "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", "val"),
        item("veh.jpg", "0 0.5 0.5 0.9 0.9\n"),
        item("nolabel.jpg", None),
        item("bad.jpg", "1 0.5 0.5 0.2\n"),
        item("corrupt.jpg", "1 0.5 0.5 0.2 0.2\n", image=False),
    ]
    kept, dropped = rf.build_tar(items, tmp_path / "o.tar")
    assert kept == {("s1", "train"): 1, ("s1", "val"): 1}
    assert {(n, r) for _, _, n, r in dropped} == {
        ("veh.jpg", "no_plate"), ("nolabel.jpg", "missing_label"), ("bad.jpg", "bad_label:bad_field_count"), ("corrupt.jpg", "corrupt_image")}
    with tarfile.open(tmp_path / "o.tar") as t:
        assert sorted(t.getnames()) == sorted(["images/train/s1__ok.jpg", "labels/train/s1__ok.txt", "images/val/s1__poly.jpg", "labels/val/s1__poly.txt", "data.yaml"])
        assert t.extractfile("labels/train/s1__ok.txt").read() == b"0 0.500000 0.500000 0.200000 0.200000\n"
        assert t.extractfile("labels/val/s1__poly.txt").read() == b"0 0.200000 0.200000 0.200000 0.200000\n"
        assert b"0: plate" in t.extractfile("data.yaml").read()
    with pytest.raises(ValueError):  # same set + name twice
        rf.build_tar([items[0], items[0]], tmp_path / "dup.tar")
