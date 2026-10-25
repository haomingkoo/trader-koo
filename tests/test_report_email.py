from __future__ import annotations

import pytest

from trader_koo.report_email import EMAIL_SETUP_ACTION_MAX_CHARS, EMAIL_SETUP_OBS_MAX_CHARS, _compact_copy

MSFT_OBS = (
    "MSFT shows bullish continuation potential with a failed breakdown back above support "
    "and a rising 50-day average."
)
MSFT_ACTION = "Bullish setup is present, but wait for confirmation instead of buying the first bounce."


@pytest.mark.parametrize("max_chars", [20, 40, EMAIL_SETUP_ACTION_MAX_CHARS, EMAIL_SETUP_OBS_MAX_CHARS, 110])
def test_compact_copy_over_limit_ends_on_whole_word_with_ellipsis(max_chars: int) -> None:
    text = " ".join(f"word{i}" for i in range(60))

    result = _compact_copy(text, max_chars=max_chars)

    assert result.endswith("…")
    assert len(result) <= max_chars + 1
    assert result[:-1].split(" ")[-1] in text.split(" ")


@pytest.mark.parametrize(
    ("text", "max_chars", "expected"),
    [
        (MSFT_OBS, 72, "MSFT shows bullish continuation potential with a failed breakdown back…"),
        (MSFT_ACTION, 64, "Bullish setup is present, but wait for confirmation instead of…"),
    ],
)
def test_compact_copy_msft_setup_text_is_not_cut_mid_word(text: str, max_chars: int, expected: str) -> None:
    assert _compact_copy(text, max_sentences=1, max_chars=max_chars) == expected


def test_compact_copy_under_limit_is_unchanged() -> None:
    assert _compact_copy("Short and sweet.", max_chars=64) == "Short and sweet."


def test_compact_copy_exact_limit_is_unchanged() -> None:
    text = "a" * 10

    assert _compact_copy(text, max_chars=10) == text


def test_compact_copy_word_ending_exactly_at_limit_is_kept() -> None:
    assert _compact_copy("alpha beta gamma", max_chars=10) == "alpha beta…"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_compact_copy_empty_returns_dash(value: object) -> None:
    assert _compact_copy(value) == "-"


def test_compact_copy_without_spaces_hard_cuts() -> None:
    assert _compact_copy("x" * 50, max_chars=10) == "x" * 10 + "…"


def test_compact_copy_strips_trailing_punctuation_before_ellipsis() -> None:
    assert _compact_copy("alpha beta, gamma delta", max_chars=12) == "alpha beta…"


# --- owner decisions: caps, mover column, board sizes -------------------------

import re  # noqa: E402

from trader_koo import report_email  # noqa: E402
from trader_koo.report_email import build_report_email_bodies  # noqa: E402


def _report(**signals: object) -> dict:
    return {"ok": True, "generated_ts": "2026-10-23T22:00:00Z", "meta": {}, "signals": dict(signals)}


def _third_column_cells(html: str) -> list[str]:
    return re.findall(r'padding:8px 0 8px 12px;[^>]*>([^<]*)</td>', html)


def test_setup_card_caps_match_owner_decision() -> None:
    assert EMAIL_SETUP_OBS_MAX_CHARS == 160
    assert EMAIL_SETUP_ACTION_MAX_CHARS == 120


def test_board_sizes_match_owner_decision() -> None:
    assert report_email.EMAIL_SETUP_ROWS_PER_BUCKET == 3
    assert report_email.EMAIL_CATALYST_ROWS_PER_SESSION == 2
    assert report_email.EMAIL_CATALYST_DAY_LIMIT == 2


def test_mover_column_shows_distance_when_not_near_extreme() -> None:
    up = [
        {"ticker": "UPA", "pct_change": 3.0, "near_52w_high": True, "pct_from_high": 1.2},
        {"ticker": "UPB", "pct_change": 2.0, "near_52w_high": False, "pct_from_high": 12.34},
        {"ticker": "UPC", "pct_change": 1.0, "near_52w_high": False, "pct_from_high": None},
    ]
    down = [
        {"ticker": "DNA", "pct_change": -3.0, "near_52w_low": True, "pct_from_low": 0.5},
        {"ticker": "DNB", "pct_change": -2.0, "near_52w_low": False, "pct_from_low": 7.5},
        {"ticker": "DNC", "pct_change": -1.0, "near_52w_low": False, "pct_from_low": None},
    ]

    _, html = build_report_email_bodies(_report(movers_up_today=up, movers_down_today=down), "")

    assert _third_column_cells(html) == [
        "near 52W high", "12.3% below 52W high", "",
        "near 52W low", "7.5% above 52W low", "",
    ]


def test_mover_text_body_has_no_third_column_dash() -> None:
    up = [{"ticker": "UPB", "pct_change": 2.0, "near_52w_high": False, "pct_from_high": 12.34}]

    text, _ = build_report_email_bodies(_report(movers_up_today=up), "")

    assert "Top gainer: UPB +2.00%" in text
    assert "52W" not in text


def _setup(ticker: str, recency: str = "fresh") -> dict:
    return {"ticker": ticker, "score": 80, "setup_tier": "A", "yolo_recency": recency, "yolo_pattern": "x"}


def test_setup_bucket_holds_three_rows() -> None:
    rows = [_setup(t) for t in ("AAA", "BBB", "CCC", "DDD")]

    sections = report_email._setup_bucket_sections(rows)

    assert [r["ticker"] for r in sections[0][2]] == ["AAA", "BBB", "CCC"]


def test_catalyst_board_shows_two_rows_and_correct_footer() -> None:
    rows = [{"ticker": t, "recommendation_state": "watch"} for t in ("AAA", "BBB", "CCC", "DDD")]
    groups = [{"display_date": "Mon", "count": 4, "sessions": [{"code": "AMC", "label": "After close", "rows": rows}]}]

    html = report_email._catalyst_board_html(groups)

    assert "Showing 2 of 4 names for this session." in html
    assert "AAA" in html and "BBB" in html and "CCC" not in html


def test_catalyst_board_omits_footer_when_all_rows_shown() -> None:
    rows = [{"ticker": t, "recommendation_state": "watch"} for t in ("AAA", "BBB")]
    groups = [{"display_date": "Mon", "count": 2, "sessions": [{"code": "AMC", "label": "After close", "rows": rows}]}]

    assert "Showing" not in report_email._catalyst_board_html(groups)
