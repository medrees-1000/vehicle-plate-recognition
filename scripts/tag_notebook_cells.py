"""Add the cell tag `remove_output` to the two image cells of notebooks/02_ccpd_yolo_check.ipynb.

Why: before committing, `nbconvert` deletes the outputs of cells tagged `remove_output`, so the
CCPD photos never reach the public repo while text and table outputs stay.

What it does, in plain words:
  1. Finds the cell whose code contains each marker line below. It must find exactly ONE cell per
     marker; with none or several it stops and changes nothing.
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
MARKERS = (
    "# vertex plot:",  # step 6b plot
    "fig.suptitle('CCPD converted YOLO boxes (red)')",  # step 6 grid
)


def dump(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False)


def tag_cells(nb: dict) -> list[int]:
    """Tag the one code cell per marker. Returns the tagged indices. Raises ValueError if a marker matches 0 or >1 cells."""
    found = []
    for marker in MARKERS:
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
    try:
        found = tag_cells(nb)
    except ValueError as e:
        raise SystemExit(f"{p}: {e}; nothing changed") from e
    p.write_text(dump(nb) + end, encoding="utf-8")
    print(f"tagged '{TAG}' on cell indices: {found}")


def selfcheck() -> None:
    def cell(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "source": [src], "outputs": []}

    ok = {"cells": [cell(MARKERS[0]), cell("x"), cell(MARKERS[1])]}
    assert tag_cells(ok) == [0, 2] and ok["cells"][1]["metadata"] == {}
    assert tag_cells(ok) == [0, 2] and ok["cells"][0]["metadata"]["tags"] == [TAG]  # running twice adds nothing
    for bad in ({"cells": [cell(MARKERS[0])]}, {"cells": [cell(MARKERS[0]), cell(MARKERS[0]), cell(MARKERS[1])]}):
        try:
            tag_cells(bad)
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
