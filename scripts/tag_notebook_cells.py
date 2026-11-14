"""Add the cell tag `remove_output` to the cells named in MARKERS for a notebook (02: two image cells; 03: three image cells and the README-printing cell; 04: three image cells; 05: the zip-listing and JSON-structure cells).

Why: before committing, `nbconvert` deletes the outputs of cells tagged `remove_output`, so the
CCPD photos never reach the public repo while text and table outputs stay.

What it does, in plain words:
  1. Finds the cell whose code contains each marker of that notebook (MARKERS is keyed by file name;
     an unknown notebook stops the script). It must find exactly ONE cell per marker; with none or
     several it stops and changes nothing.
  2. Adds the tag to that cell and prints the cell's index (0 = first cell).
  3. Rewrites the file only if the file's JSON formatting is the standard one (1-space indent), so
     `git diff` shows just the tag lines. If the formatting would change, it stops instead.

Usage:  python scripts/tag_notebook_cells.py notebooks/02_ccpd_yolo_check.ipynb
Check:  python scripts/tag_notebook_cells.py --selfcheck
"""
import json
import sys
from pathlib import Path

TAG = "remove_output"
MARKERS = {
    "02_ccpd_yolo_check.ipynb": (
        "# vertex plot:",  # step 6b plot
        "fig.suptitle('CCPD converted YOLO boxes (red)')",  # step 6 grid
    ),
    "03_roboflow_audit.ipynb": (
        "import yaml  # preinstalled on Colab (PyYAML)",  # step 1: prints the READMEs, re-runnable any time
        "def sheet(items, title, out, ncols=5):",  # step 6: random contact sheets
        "orb_fl = sorted(",  # step 6: flagged-pair sheets
        "TOP_SET, TOP_N = 'usa_v1', 10",  # step 6b: top random pairs
    ),
    "04_us_split.ipynb": (
        "# evidence sheet: 20 highest test-vs-pool scores",  # Roboflow images side by side
        "# visual box check",  # sample images from the tar with boxes
        "# contact sheet: removed name groups larger than 20",  # big removed groups, Roboflow images
    ),
    "05_uc3m_check.ipynb": (
        "# Cell A: zip listing",  # prints folder names, file names and README text from the UC3M-LP zip
        "# Cell C: JSON structure",  # key paths and label patterns (no plate text, but the label format)
        "# visual box check: UC3M-LP",  # sample images from the converted tar with boxes
    ),
}


def dump(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False)


def tag_cells(nb: dict, markers: tuple[str, ...]) -> list[int]:
    """Tag the one code cell per marker. Returns the tagged indices. Raises ValueError if a marker matches 0 or >1 cells."""
    found = []
    for marker in markers:
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and marker in "".join(c["source"])]
        if len(hits) != 1:
            raise ValueError(f"marker {marker!r} matches {len(hits)} cells (cell indices {hits}), need exactly 1")
        found.append(hits[0])
    for i in found:
        tags = nb["cells"][i].setdefault("metadata", {}).setdefault("tags", [])
        if TAG not in tags:
            tags.append(TAG)
    return found


def main(path: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    nb = json.loads(text)
    end = "\n" if text.endswith("\n") else ""  # keep the file's own ending, so git diff shows no extra line
    if dump(nb) + end != text:
        raise SystemExit(f"{p}: JSON formatting is not the standard 1-space indent; refusing to rewrite it")
    if p.name not in MARKERS:
        raise SystemExit(f"{p.name}: no markers defined for this notebook (known: {sorted(MARKERS)}); nothing changed")
    try:
        found = tag_cells(nb, MARKERS[p.name])
    except ValueError as e:
        raise SystemExit(f"{p}: {e}; nothing changed") from e
    p.write_text(dump(nb) + end, encoding="utf-8")
    print(f"tagged '{TAG}' on cell indices: {found}")


def selfcheck() -> None:
    def cell(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "source": [src], "outputs": []}

    m = MARKERS["02_ccpd_yolo_check.ipynb"]
    ok = {"cells": [cell(m[0]), cell("x"), cell(m[1])]}
    assert tag_cells(ok, m) == [0, 2] and ok["cells"][1]["metadata"] == {}
    assert tag_cells(ok, m) == [0, 2] and ok["cells"][0]["metadata"]["tags"] == [TAG]  # running twice adds nothing
    for bad in ({"cells": [cell(m[0])]}, {"cells": [cell(m[0]), cell(m[0]), cell(m[1])]}):
        try:
            tag_cells(bad, m)
        except ValueError:
            continue
        raise AssertionError("zero or two matches must raise")
    print("selfcheck ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]:
        selfcheck()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(__doc__)
