"""Focused coverage for Green Barrier report serialization."""
from __future__ import annotations

from trader_koo.report.serializer import to_markdown


def test_to_markdown_includes_green_barrier_conditions_and_coverage() -> None:
    report = {
        "counts": {},
        "latest_data": {},
        "signals": {
            "green_barrier_hits": [
                {
                    "ticker": "HIT",
                    "timeframe": "monthly",
                    "value": -99.0,
                    "threshold": -95.0,
                    "distance_to_barrier": 1.0,
                    "asof": "2026-08-13",
                    "close": 100.0,
                    "in_progress_value": -80.5,
                    "in_progress_asof": "2026-08-20",
                }
            ],
            "green_barrier_coverage": {
                "scan_asof": "2026-08-14",
                "threshold": -95.0,
                "max_age_days": 7,
                "source_ticker_count": 2,
                "scanned_ticker_count": 1,
                "stale_skipped_count": 1,
                "stale_skipped_tickers": ["STALE"],
                "invalid_date_skipped_count": 0,
                "insufficient_history_skipped_count": 5,
            },
        },
    }

    text = to_markdown(report)

    assert "## Green Barrier Current Conditions" in text
    assert "Repeated daily while active" in text
    assert "| now (bar forming) | now as of |" in text
    assert "| 100.0 | -80.5 | 2026-08-20 |" in text
    assert "### Green Barrier Scan Coverage" in text
    assert "**stale_skipped_tickers**: STALE" in text
    assert "**insufficient_history_skipped_count**: 5" in text
