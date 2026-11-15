"""Tests for alpr.metrics. Synthetic strings only; expectations computed by hand."""
import pytest

from alpr import metrics as m


def test_normalise_rules():
    assert m.normalise("AbC") == "abc"                  # case
    assert m.normalise("a b\tc") == "abc"               # spaces
    assert m.normalise("a-b") == "ab"                   # hyphen
    assert m.normalise("a.b") == "ab"                   # dot
    assert m.normalise("a·b") == "ab"              # middle dot
    assert m.normalise("a:b") == "ab"                   # colon
    assert m.normalise("Éñ") == "en"          # accents stripped
    assert m.normalise("a*b") == "a*b"                  # wildcard kept


def test_exact_match():
    assert m.exact_match("ab 12", "AB-12")
    assert not m.exact_match("ab12", "ab13")
    assert not m.exact_match("", "ab12")                # empty prediction is wrong


def test_edit_distance_and_cer_single_edits():
    assert m.edit_distance("kitten", "sitting") == 3
    assert m.cer("ABX4", "ABC4") == 0.25                # substitution: 1 / 4
    assert m.cer("ABCC4", "ABC4") == 0.25               # insertion
    assert m.cer("AB4", "ABC4") == 0.25                 # deletion
    assert m.cer("ab-c4", "ABC4") == 0.0                # normalised first


def test_empty_prediction_costs_reference_length():
    assert m.edit_distance("", "ABC") == 3
    assert m.cer("", "ABC") == 1.0


def test_micro_differs_from_macro():
    pairs = [("AB", "AC"), ("XYZW", "XYZW")]            # edits 1 of 2 chars, 0 of 4 chars
    micro = m.aggregate_cer(pairs)
    macro = sum(m.cer(p, r) for p, r in pairs) / 2
    assert micro == pytest.approx(1 / 6)
    assert macro == pytest.approx(0.25)
    assert micro != macro


def test_wildcard_match():
    ref = "AD-1*3*"                                      # text "1*3*"
    assert m.wildcard_match("1234", ref)                # each * accepts one character
    assert m.wildcard_match("1x3y", ref)
    assert m.wildcard_match("12 34", ref)               # spaces normalised away
    assert not m.wildcard_match("2234", ref)            # a visible position differs
    assert not m.wildcard_match("1243", ref)            # wrong character at a visible position (index 2)
    assert not m.wildcard_match("123", ref)             # too short
    assert not m.wildcard_match("12345", ref)           # too long
    assert not m.wildcard_match("AD1234", ref)          # the prefix is not part of the text
    assert m.wildcard_match("1234", "ZZ-1*3*")          # a different prefix changes nothing
    assert m.wildcard_match("ab", "ab")                 # no '-' : all of it is text


def test_errors_do_not_echo_inputs():
    with pytest.raises(ValueError) as e:
        m.cer("zq1", "-.-")
    assert "zq1" not in str(e.value) and "-.-" not in str(e.value)
    with pytest.raises(ValueError) as e:
        m.exact_match("zq1", " : ")
    assert "zq1" not in str(e.value)
    with pytest.raises(ValueError) as e:
        m.wildcard_match("zq1", "AD-.-")
    assert "zq1" not in str(e.value) and "AD" not in str(e.value)
    with pytest.raises(ValueError):
        m.aggregate_cer([])
    bad = [{"pred": "zq1", "proposed_text": "QQ77", "corrected_text": "", "final_status": "weird-status", "reason": ""}]
    with pytest.raises(ValueError) as e:
        m.score_us(bad)
    assert "zq1" not in str(e.value) and "QQ77" not in str(e.value) and "weird-status" not in str(e.value)
    bad[0]["final_status"], bad[0]["proposed_text"] = "readable", " - "
    with pytest.raises(ValueError) as e:
        m.score_us(bad)
    assert "zq1" not in str(e.value)


def _row(status, ref, pred, reason="", corrected=""):
    return {"pred": pred, "proposed_text": ref, "corrected_text": corrected, "final_status": status, "reason": reason}


def test_score_us_groups():
    rows = [
        _row("readable", "AAA111", "AAA111"),                           # right
        _row("readable", "BBB2", "BBB3", reason="partial"),             # partial, 1 substitution
        _row("readable", "CCC333", "CCC334", corrected="CCC334"),       # corrected_text wins over proposed_text; right
        _row("readable", "DDD444", "DDD44"),                            # 1 deletion
        _row("guess", "EEE555", "EEE555"),                              # right
        _row("guess", "FFF666", "FFF667"),                              # 1 substitution
        _row("unreadable", "", "zzz", reason="too_small"),              # not scored
        _row("not_plate", "", "zzz"),                                   # not scored
        _row("", "", "zzz"),                                            # not reviewed: not scored
    ]
    s = m.score_us(rows)
    assert s["excluded"] == {"unreadable": 1, "not_plate": 1, "pending": 1}
    c, p, g, pa = s["confirmed"], s["confirmed_no_partial"], s["confirmed_plus_guess"], s["partial"]
    assert (c["n"], p["n"], pa["n"], g["n"]) == (4, 3, 1, 6)
    assert c["n"] == p["n"] + pa["n"]                                   # groups add up as declared
    assert (c["exact"], c["exact_rate"], c["cer"]) == (2, 0.5, pytest.approx(2 / 22))
    assert (p["exact"], p["exact_rate"], p["cer"]) == (2, pytest.approx(2 / 3), pytest.approx(1 / 18))
    assert (pa["exact"], pa["exact_rate"], pa["cer"]) == (0, 0.0, 0.25)
    assert (g["exact"], g["exact_rate"], g["cer"]) == (3, 0.5, pytest.approx(3 / 34))


def test_score_us_empty_groups_have_no_rates():
    s = m.score_us([_row("readable", "AB12", "AB12")])
    assert s["partial"] == {"n": 0, "exact": 0, "exact_rate": None, "cer": None}
