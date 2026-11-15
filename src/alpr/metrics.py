"""OCR scoring plumbing (M2): match rule, edit distance, CER, the UC3M wildcard match and the US group scores. Pure functions.

The match rule and the label statuses are those of PLAN.md (Evaluation, M1b-5); the normaliser and the prefix split are reused from
alpr.uslabels and alpr.uc3m. Error messages never contain the input strings (they are plate text).
Self-check: uv run pytest tests/test_metrics.py
"""
from collections.abc import Iterable, Mapping

from alpr import uc3m, uslabels


def normalise(text: str) -> str:
    """Case-insensitive; spaces, hyphens, dots, middle dots, colons removed; accents stripped. '*' is kept (it is a wildcard)."""
    return uslabels._compact(text)


def _ref(ref: str) -> str:
    r = normalise(ref)
    if not r:
        raise ValueError("empty reference text after normalisation")
    return r


def exact_match(pred: str, ref: str) -> bool:
    """Equal after normalise on both sides. An empty reference raises ValueError; an empty prediction is simply wrong."""
    return normalise(pred) == _ref(ref)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert, delete, substitute; each 1) of the two strings as given (no normalising)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """Character error rate = edits / len(normalised ref), both sides normalised. An empty prediction costs len(ref) edits (CER 1.0)."""
    r = _ref(ref)
    return edit_distance(normalise(pred), r) / len(r)


def aggregate_cer(pairs: Iterable[tuple[str, str]]) -> float:
    """Micro average over (pred, ref) pairs: total edits / total reference characters (normalised). Declared before any score exists:
    long plates weigh more, and it is not the mean of the per-plate CERs. No pairs raises ValueError."""
    edits = chars = 0
    for pred, ref in pairs:
        r = _ref(ref)
        edits += edit_distance(normalise(pred), r)
        chars += len(r)
    if not chars:
        raise ValueError("no pairs to aggregate")
    return edits / chars


def wildcard_match(pred: str, ref_masked: str) -> bool:
    """UC3M rule (PLAN.md, UC3M-LP rule c): compare the text after the prefix of the label (`lp_id` = prefix + '-' + text; no '-' means all of it
    is text). A '*' in the label is a masked character that accepts any one character; equal length is required. The prefix is not scored.
    `pred` and the label text are normalised first ('*' is kept). A label with no text after the prefix raises ValueError."""
    ref = _ref(uc3m._text(ref_masked))
    p = normalise(pred)
    return len(p) == len(ref) and all(c == "*" or c == q for q, c in zip(p, ref))


def _group(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    hit = sum(exact_match(p, r) for p, r in pairs)
    return {"n": n, "exact": hit, "exact_rate": hit / n if n else None, "cer": aggregate_cer(pairs) if n else None}


def score_us(rows: Iterable[Mapping[str, str]]) -> dict:
    """Scores for the US test labels (PLAN.md, US OCR scoring rule). Each row needs `pred` (the engine's text, '' if none), `proposed_text`,
    `corrected_text`, `final_status` and `reason`. The scoring text is corrected_text if not empty, otherwise proposed_text. Groups, each
    {n, exact, exact_rate, cer (micro)} with None rates when n is 0:
      confirmed            final_status 'readable' (the 65 of the real data, partial plates included)
      confirmed_no_partial confirmed without the rows whose reason is 'partial' (61)
      confirmed_plus_guess confirmed plus 'guess' (86)
      partial              the partial plates alone (4)
    unreadable, not_plate and not yet reviewed ('') rows are not scored; their counts are returned under `excluded`.
    An unknown final_status, or a scored row with no text, raises ValueError naming the row index only."""
    full, plain, partial, guess = [], [], [], []
    excluded = {"unreadable": 0, "not_plate": 0, "pending": 0}
    for i, r in enumerate(rows):
        status = r["final_status"]
        if status in ("unreadable", "not_plate"):
            excluded[status] += 1
        elif status == "":
            excluded["pending"] += 1
        elif status in ("readable", "guess"):
            pair = (r["pred"], r["corrected_text"] or r["proposed_text"])
            if not normalise(pair[1]):
                raise ValueError(f"row {i}: no scoring text")
            if status == "guess":
                guess.append(pair)
            else:
                full.append(pair)
                (partial if r["reason"] == "partial" else plain).append(pair)
        else:
            raise ValueError(f"row {i}: unknown final_status")
    return {"confirmed": _group(full), "confirmed_no_partial": _group(plain), "confirmed_plus_guess": _group(full + guess),
            "partial": _group(partial), "excluded": excluded}
