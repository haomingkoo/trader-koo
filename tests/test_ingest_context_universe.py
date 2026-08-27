import datetime as dt
import json
import sqlite3

from trader_koo.scripts.update_market_db import (
    DEFAULT_SOFT_FAIL_TICKERS,
    ensure_schema,
    get_succeeded_tickers_from_latest_run,
)


def test_vix_term_structure_uses_provider_symbols_only():
    assert "^VIX3M" in DEFAULT_SOFT_FAIL_TICKERS
    assert "^VIX6M" in DEFAULT_SOFT_FAIL_TICKERS
    assert "VIX3M" not in DEFAULT_SOFT_FAIL_TICKERS
    assert "VIX6M" not in DEFAULT_SOFT_FAIL_TICKERS


def test_resume_unions_current_day_successes_across_incremental_runs(tmp_path):
    conn = sqlite3.connect(tmp_path / "ingest.db")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    args_json = json.dumps({"use_sp500": True, "skip_price": False})
    conn.executemany(
        """INSERT INTO ingest_runs
           (run_id, started_ts, status, tickers_total, args_json)
           VALUES (?, ?, ?, ?, ?)""",
        [
            ("large-failed", f"{today}T01:00:00Z", "failed", 2, args_json),
            ("context-ok", f"{today}T02:00:00Z", "ok", 1, args_json),
        ],
    )
    conn.executemany(
        """INSERT INTO ingest_ticker_status
           (run_id, ticker, started_ts, status, price_rows)
           VALUES (?, ?, ?, 'ok', 1)""",
        [
            ("large-failed", "AAPL", f"{today}T01:00:00Z"),
            ("large-failed", "MSFT", f"{today}T01:00:00Z"),
            ("context-ok", "SPY", f"{today}T02:00:00Z"),
        ],
    )
    conn.commit()

    assert get_succeeded_tickers_from_latest_run(conn) == {"AAPL", "MSFT", "SPY"}
