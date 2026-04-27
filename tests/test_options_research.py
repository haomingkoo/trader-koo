from __future__ import annotations

import datetime as dt
import sqlite3
from types import SimpleNamespace

import pandas as pd

from trader_koo.options_research import (
    build_options_premium_proxy,
    build_options_positioning_context,
    load_options_snapshot_tickers,
    snapshot_options_iv,
)
from trader_koo.report.setup_scoring import (
    _apply_news_research_context,
    _apply_options_research_context,
    _apply_setup_eval_fields,
)
from trader_koo.rss_news import persist_rss_headline_snapshot


def _conn_with_options() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE options_iv (
            snapshot_ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            expiration TEXT NOT NULL,
            option_type TEXT NOT NULL,
            strike REAL NOT NULL,
            last_price REAL,
            bid REAL,
            ask REAL,
            implied_vol REAL,
            open_interest REAL,
            volume REAL,
            moneyness REAL,
            PRIMARY KEY (snapshot_ts, ticker, expiration, option_type, strike)
        );

        CREATE TABLE price_daily (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL
        );
        """
    )
    base = dt.date(2026, 1, 1)
    for i in range(80):
        conn.execute(
            "INSERT INTO price_daily (ticker, date, close) VALUES ('AMD', ?, ?)",
            ((base + dt.timedelta(days=i)).isoformat(), 100.0 + i * 0.25),
        )

    snapshots = [
        ("2026-03-01T00:00:00Z", 0.90, 1000, 800),
        ("2026-03-08T00:00:00Z", 0.82, 1300, 900),
        ("2026-03-15T00:00:00Z", 0.70, 1500, 1100),
        ("2026-03-22T00:00:00Z", 0.35, 5000, 1200),
    ]
    for snap, iv, call_oi, put_oi in snapshots:
        conn.execute(
            """
            INSERT INTO options_iv (
                snapshot_ts, ticker, expiration, option_type, strike,
                last_price, bid, ask, implied_vol, open_interest, volume, moneyness
            ) VALUES (?, 'AMD', '2026-05-15', 'call', 100, 4.0, 3.9, 4.1, ?, ?, 500, 1.0)
            """,
            (snap, iv, call_oi),
        )
        conn.execute(
            """
            INSERT INTO options_iv (
                snapshot_ts, ticker, expiration, option_type, strike,
                last_price, bid, ask, implied_vol, open_interest, volume, moneyness
            ) VALUES (?, 'AMD', '2026-05-15', 'put', 100, 3.5, 3.4, 3.6, ?, ?, 200, 1.0)
            """,
            (snap, iv * 1.05, put_oi),
        )
    conn.commit()
    return conn


def test_build_options_positioning_context_flags_underpriced_positioning():
    conn = _conn_with_options()

    ctx = build_options_positioning_context(conn, "AMD")

    assert ctx["available"] is True
    assert ctx["source"] == "yfinance_options_iv"
    assert ctx["iv_rank_pct"] <= 30
    assert ctx["oi_rank_pct"] >= 90
    assert ctx["signal"] == "underpriced_positioning"
    assert ctx["underpriced_score"] > 60
    assert "Not real-time sweeps" in ctx["source_note"]


def test_build_options_premium_proxy_estimates_call_minus_put_premium():
    conn = _conn_with_options()

    payload = build_options_premium_proxy(conn, limit=10)

    assert payload["available"] is True
    assert payload["source"] == "yfinance_options_iv"
    assert "not signed" in payload["premium_proxy_note"]
    assert payload["count"] == 1
    row = payload["rows"][0]
    assert row["ticker"] == "AMD"
    assert row["snapshot_ts"] == "2026-03-22T00:00:00Z"
    assert row["call_volume_premium"] == 200000.0
    assert row["put_volume_premium"] == 70000.0
    assert row["net_volume_premium"] == 130000.0
    assert row["premium_bias"] == "call_premium_skew"
    assert row["primary_premium_source"] == "volume"
    assert row["net_oi_premium"] == 1580000.0
    assert row["total_volume"] == 700.0
    assert row["volume_skew_pct"] == 48.1
    assert row["oi_skew_pct"] == 65.3
    assert row["volume_rank_pct"] == 50.0
    assert row["smart_score"] == 49.9
    assert row["smart_signal"] == "watch"
    assert "strong_flow" in row["smart_tags"]
    assert "call_oi_lead" in row["smart_tags"]


def test_apply_options_research_context_adds_setup_fields():
    conn = _conn_with_options()
    rows = [{"ticker": "AMD", "score": 72.0, "setup_tier": "B", "signal_bias": "bullish", "components": {}}]

    annotated = _apply_options_research_context(conn, rows)

    assert annotated == 1
    assert rows[0]["options_positioning_signal"] == "underpriced_positioning"
    assert rows[0]["options_oi_rank_pct"] >= 90
    assert rows[0]["score"] > 72.0
    assert rows[0]["components"]["options"] > 0


def test_apply_news_research_context_sanitizes_and_scores(monkeypatch):
    def _fake_news(**kwargs):
        return {
            "provider": "rss_aggregator",
            "available": True,
            "article_count": 2,
            "headlines": [
                {
                    "feed_ticker": "AMD",
                    "title": "<b>AMD rallies on strong demand</b>",
                    "source": "Example",
                    "score": 78,
                    "label": "Greed",
                    "macro_relevant": False,
                },
                {
                    "feed_ticker": None,
                    "title": "Fed rate risk pressures markets",
                    "source": "Macro",
                    "score": 35,
                    "label": "Fear",
                    "macro_relevant": True,
                },
            ],
        }

    monkeypatch.setattr("trader_koo.rss_news.fetch_rss_headlines", _fake_news)
    rows = [{"ticker": "AMD", "score": 70.0, "setup_tier": "B", "signal_bias": "bullish", "components": {}}]

    meta = _apply_news_research_context(rows)

    assert meta["annotated"] == 1
    assert rows[0]["news_sentiment_score"] == 78
    assert rows[0]["macro_news_score"] == 35
    assert "<" not in rows[0]["news_context"]["ticker_headlines"][0]["title"]
    assert rows[0]["components"]["news"] > 0


def test_apply_news_research_context_skips_historical_live_rss():
    rows = [{"ticker": "AMD", "score": 70.0, "setup_tier": "B", "signal_bias": "bullish", "components": {}}]

    meta = _apply_news_research_context(rows, as_of_date="2024-01-05")

    assert meta["annotated"] == 0
    assert meta["skipped_reason"] == "live_rss_not_point_in_time"
    assert "news_context" not in rows[0]


def test_apply_news_research_context_uses_historical_rss_snapshot():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    persist_rss_headline_snapshot(
        conn,
        {
            "provider": "rss_aggregator",
            "available": True,
            "headlines": [
                {
                    "feed_ticker": "AMD",
                    "title": "AMD rallies on strong demand",
                    "source": "Yahoo Finance",
                    "score": 78,
                    "label": "Greed",
                    "time_published": "2024-01-05T12:00:00Z",
                    "macro_relevant": False,
                },
                {
                    "feed_ticker": None,
                    "title": "Fed rate risk pressures markets",
                    "source": "Macro",
                    "score": 35,
                    "label": "Fear",
                    "time_published": "2024-01-05T13:00:00Z",
                    "macro_relevant": True,
                },
            ],
        },
        snapshot_date="2024-01-05",
        snapshot_ts="2024-01-05T14:00:00Z",
    )
    conn.commit()
    rows = [{"ticker": "AMD", "score": 70.0, "setup_tier": "B", "signal_bias": "bullish", "components": {}}]

    meta = _apply_news_research_context(rows, as_of_date="2024-01-05", conn=conn)

    assert meta["annotated"] == 1
    assert meta["provider"] == "rss_snapshot"
    assert rows[0]["news_sentiment_score"] == 78
    assert rows[0]["macro_news_score"] == 35
    assert rows[0]["components"]["news"] > 0


def test_setup_eval_fields_adds_continuous_calibrated_probability():
    rows = [
        {
            "ticker": "AMD",
            "score": 72.0,
            "setup_family": "bullish_reversal",
            "signal_bias": "bullish",
            "setup_tier": "B",
        }
    ]
    reliability = {
        ("bullish_reversal", "long"): {
            "hit_rate_pct": 70.0,
            "calls": 20,
            "avg_signed_return_pct": 1.2,
        }
    }

    _apply_setup_eval_fields(
        rows,
        reliability_lookup=reliability,
        min_sample=5,
        hit_threshold_pct=55.0,
        baseline_hit_rate_pct=50.0,
    )

    assert rows[0]["calibrated_hit_prob"] > 0.6
    assert rows[0]["probability_source"] == "empirical_bayes_setup_eval"
    assert rows[0]["probability_sample_size"] == 20
    assert rows[0]["probability_label"] == "high"


class _FakeOptionTicker:
    options = ["2026-05-15", "2026-06-19"]

    def __init__(self, ticker: str):
        self.ticker = ticker

    def history(self, period: str = "5d"):
        return pd.DataFrame({"Close": [99.0, 100.0]})

    def option_chain(self, expiration: str):
        calls = pd.DataFrame(
            [
                {
                    "strike": 100,
                    "lastPrice": 4.2,
                    "bid": 4.1,
                    "ask": 4.3,
                    "impliedVolatility": 0.42,
                    "openInterest": 1200,
                    "volume": 300,
                },
                {
                    "strike": 160,
                    "lastPrice": 0.4,
                    "bid": 0.3,
                    "ask": 0.5,
                    "impliedVolatility": 0.9,
                    "openInterest": 20,
                    "volume": 5,
                },
            ]
        )
        puts = pd.DataFrame(
            [
                {
                    "strike": 95,
                    "lastPrice": 3.8,
                    "bid": 3.7,
                    "ask": 3.9,
                    "impliedVolatility": 0.45,
                    "openInterest": 800,
                    "volume": 120,
                }
            ]
        )
        return SimpleNamespace(calls=calls, puts=puts)


def test_snapshot_options_iv_fetches_and_writes_bounded_yfinance_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    summary = snapshot_options_iv(
        conn,
        ["AMD", "^VIX"],
        snapshot_ts="2026-04-24T22:00:00Z",
        max_expiries=1,
        min_moneyness=0.9,
        max_moneyness=1.1,
        ticker_factory=_FakeOptionTicker,
    )

    assert summary["tickers_total"] == 1
    assert summary["tickers_refreshed"] == 1
    assert summary["rows_inserted"] == 2
    rows = conn.execute(
        "SELECT ticker, expiration, option_type, strike, implied_vol, open_interest, moneyness "
        "FROM options_iv ORDER BY option_type"
    ).fetchall()
    assert [row["option_type"] for row in rows] == ["call", "put"]
    assert all(row["ticker"] == "AMD" for row in rows)
    assert rows[0]["expiration"] == "2026-05-15"
    assert rows[0]["moneyness"] == 1.0


def test_snapshot_options_iv_skips_recent_snapshots_without_force():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    snapshot_options_iv(
        conn,
        ["AMD"],
        snapshot_ts="2026-04-24T22:00:00Z",
        max_expiries=1,
        ticker_factory=_FakeOptionTicker,
    )

    summary = snapshot_options_iv(
        conn,
        ["AMD"],
        snapshot_ts="2026-04-25T01:00:00Z",
        min_interval_hours=20,
        ticker_factory=_FakeOptionTicker,
    )

    assert summary["tickers_refreshed"] == 0
    assert summary["tickers_skipped_recent"] == 1
    assert summary["rows_inserted"] == 0


def test_load_options_snapshot_tickers_prefers_latest_report_then_defaults(tmp_path):
    report = tmp_path / "daily_report_latest.json"
    report.write_text(
        """
        {
          "signals": {
            "suggestions": {"items": [{"ticker": "nvda"}]},
            "setup_quality_top": [{"ticker": "AMD"}, {"ticker": "^VIX"}, {"ticker": "MSFT"}]
          }
        }
        """,
        encoding="utf-8",
    )

    tickers = load_options_snapshot_tickers(
        latest_report_path=report,
        max_tickers=4,
    )

    assert tickers == ["NVDA", "AMD", "MSFT", "SPY"]
