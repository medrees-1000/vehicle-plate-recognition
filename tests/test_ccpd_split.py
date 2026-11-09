import random
from collections import Counter, defaultdict

import pytest

from alpr import ccpd

N_PER = 40  # images per label


def _name(plate_no: int, uniq: int, gap: bool = False) -> str:
    plate = "_".join(map(str, (0, 0, *(plate_no // 30**e % 30 for e in (3, 2, 1, 0)), 0)))
    verts = "80&40_10&40_10&10_80&10" if gap else "50&40_10&40_10&10_50&10"
    return f"025-0_0-10&10_50&40-{verts}-{plate}-{uniq}-15.jpg"  # uniq (brightness field) keeps names distinct


def _data() -> tuple[list[ccpd.Row], dict[str, int]]:
    rng = random.Random(1)
    plate_no = list(range(len(ccpd.SUBSETS) * N_PER))
    for k in range(1, len(plate_no)):
        if k % 40 == 0 or k % 7 == 3:  # repeated plates, some across labels
            plate_no[k] = plate_no[k - 1]
    rows, hashes = [], {}
    for k, pn in enumerate(plate_no):
        subset = ccpd.SUBSETS[k // N_PER]
        name = _name(pn, k, gap=subset == "ccpd_base" and k % 4 == 0)
        rows.append((f"{subset}/{name}", subset, "train"))
        hashes[name] = rng.getrandbits(64)
    for k in range(0, len(rows) - 1, 10):  # near-duplicates (2 bits apart) on other plates
        hashes[rows[k + 1][0].split("/")[1]] = hashes[rows[k][0].split("/")[1]] ^ 0b101
    return rows, hashes


def test_split() -> None:
    rows, hashes = _data()
    out = ccpd.split_by_plate(rows, hashes, heldout=16, val=40)
    side = Counter(r[2] for r in out)
    assert side["heldout"] == 16 and side["val"] == 40 and sum(side.values()) == len(rows) == len(out)
    for split, n in (("heldout", 2), ("val", 5)):
        assert set(Counter(r[1] for r in out if r[2] == split).values()) == {n}
    for col in (3, 4):  # plate, bundle: each on one side only
        where = defaultdict(set)
        for r in out:
            where[r[col]].add(r[2])
        assert all(len(v) == 1 for v in where.values())
    by_name = {r[0].split("/")[1]: r[2] for r in out}
    close = [(a, b) for a in hashes for b in hashes if a < b and bin(hashes[a] ^ hashes[b]).count("1") <= 6]
    assert close and all(by_name[a] == by_name[b] for a, b in close)
    gaps = [r for r in out if r[5]]
    assert gaps and all(r[2] != "heldout" for r in gaps)
    assert out == ccpd.split_by_plate(rows, hashes, heldout=16, val=40)  # same seed, same result


def test_split_errors() -> None:
    rows, hashes = _data()
    with pytest.raises(ValueError, match="no dHash"):
        ccpd.split_by_plate(rows, {}, heldout=16, val=40)
    with pytest.raises(ValueError, match="bundle"):  # equal hashes: one giant bundle
        ccpd.split_by_plate(rows, dict.fromkeys(hashes, 0), heldout=16, val=40)
