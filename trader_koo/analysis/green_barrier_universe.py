"""The ingest universe the Green Barrier scan is responsible for."""
from __future__ import annotations

import logging
import sqlite3

LOG = logging.getLogger("trader_koo.analysis.green_barrier")


def current_ingest_universe(conn: sqlite3.Connection) -> tuple[set[str] | None, str]:
    """Return the tickers the latest ingest run covered, and where that came from.

    ``price_daily`` is append-only, so it is not a universe: it accumulates every
    symbol ever ingested, including delisted and index-rotated ones. Those cannot
    go fresh again, so scanning them guarantees a staleness warning forever. The
    ingest ledger is the only record of what the pipeline is currently
    responsible for.

    Returns ``(None, "price_daily")`` when the ledger is unavailable, which keeps
    the conservative behaviour of scanning everything. That still raises the
    coverage warning and blocks publication, whereas scanning nothing would
    publish a clean report on no evidence.
    """
    try:
        row = conn.execute(
            "SELECT run_id FROM ingest_runs ORDER BY started_ts DESC LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return None, "price_daily"
        run_id = str(row[0])
        tickers = {
            str(record[0]).upper()
            for record in conn.execute(
                "SELECT DISTINCT ticker FROM ingest_ticker_status WHERE run_id = ?",
                (run_id,),
            )
            if record[0]
        }
    except sqlite3.Error as exc:
        LOG.warning(
            "Ingest universe unavailable (%s); scanning every ticker in price_daily", exc
        )
        return None, "price_daily"

    if not tickers:
        LOG.warning(
            "Ingest run %s recorded no tickers; scanning every ticker in price_daily",
            run_id,
        )
        return None, "price_daily"
    return tickers, f"ingest_runs:{run_id}"
