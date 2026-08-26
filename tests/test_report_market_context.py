from __future__ import annotations

import datetime as dt
import sqlite3
from unittest.mock import patch

from trader_koo.report.market_context import (
    _build_regime_context,
    _build_regime_llm_commentary,
)


def test_regime_context_does_not_invent_a_source_without_price_table() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        payload = _build_regime_context(conn)
    finally:
        conn.close()

    assert payload["source"] == "unavailable"


def test_regime_context_requires_enough_vix_history_for_price_source(mem_conn) -> None:
    start = dt.date(2026, 6, 1)
    mem_conn.executemany(
        """
        INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
        VALUES ('^VIX', ?, 15, 16, 14, 15, 1000)
        """,
        [((start + dt.timedelta(days=index)).isoformat(),) for index in range(54)],
    )
    mem_conn.commit()

    payload = _build_regime_context(mem_conn)

    assert payload["source"] == "unavailable"
    assert payload["asof_date"] is None


def test_regime_context_names_price_source_only_with_usable_vix(mem_conn) -> None:
    start = dt.date(2026, 6, 1)
    mem_conn.executemany(
        """
        INSERT INTO price_daily (ticker, date, open, high, low, close, volume)
        VALUES ('^VIX', ?, ?, ?, ?, ?, 1000)
        """,
        [
            (
                (start + dt.timedelta(days=index)).isoformat(),
                14.0 + index / 100,
                16.0 + index / 100,
                13.0 + index / 100,
                15.0 + index / 100,
            )
            for index in range(60)
        ],
    )
    mem_conn.commit()

    payload = _build_regime_context(mem_conn)

    assert payload["source"] == "price_daily:^VIX"
    assert payload["asof_date"] == (start + dt.timedelta(days=59)).isoformat()


def test_regime_commentary_marks_deterministic_copy_as_rule() -> None:
    with patch("trader_koo.llm_narrative.llm_enabled", return_value=False):
        payload = _build_regime_llm_commentary({"summary": "Mixed regime."})

    assert payload["source"] == "rule"


def test_regime_commentary_marks_accepted_model_copy_as_llm() -> None:
    rewritten = {
        "observation": "Model observation.",
        "action": "Model action.",
        "risk_note": "Model risk.",
    }
    with (
        patch("trader_koo.llm_narrative.llm_enabled", return_value=True),
        patch(
            "trader_koo.llm_narrative.maybe_rewrite_setup_copy",
            return_value=rewritten,
        ),
    ):
        payload = _build_regime_llm_commentary({"summary": "Mixed regime."})

    assert payload["source"] == "llm"
    assert payload["observation"] == "Model observation."


def test_regime_commentary_does_not_label_nonempty_baseline_fallback_as_llm() -> None:
    baseline = {
        "observation": "Mixed regime.",
        "action": "No urgency.",
        "risk_note": "context_only_signal",
    }
    with (
        patch("trader_koo.llm_narrative.llm_enabled", return_value=True),
        patch(
            "trader_koo.llm_narrative.maybe_rewrite_setup_copy",
            return_value=baseline,
        ),
    ):
        payload = _build_regime_llm_commentary({"summary": "Mixed regime."})

    assert payload["source"] == "rule"
