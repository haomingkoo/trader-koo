"""Coverage statistics over a loaded sec_fundamentals store."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from trader_koo.fundamentals import config as cfg


def _sectors(universe: sqlite3.Connection) -> dict[str, str]:
    rows = universe.execute(
        """SELECT ticker, raw_json FROM finviz_fundamentals f
           WHERE snapshot_ts = (SELECT MAX(snapshot_ts) FROM finviz_fundamentals WHERE ticker = f.ticker)"""
    )
    out: dict[str, str] = {}
    for ticker, raw in rows:
        try:
            out[ticker.upper()] = json.loads(raw).get("Sector") or "Unknown"
        except (TypeError, ValueError):
            out[ticker.upper()] = "Unknown"
    return out


def coverage_report(
    conn: sqlite3.Connection,
    universe: sqlite3.Connection,
    sectors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """``sectors`` overrides the Finviz sector in the universe DB, which is often empty."""
    sectors = sectors or _sectors(universe)
    concept_of = {tag: name for name, pairs in cfg.CONCEPTS.items() for _, tag in pairs}
    have: dict[int, set[str]] = defaultdict(set)
    for cik, tag in conn.execute("SELECT DISTINCT cik, tag FROM sec_fundamentals"):
        have[cik].add(concept_of[tag])
    per_sector: dict[str, dict[str, Any]] = defaultdict(lambda: {"companies": 0, "concepts": defaultdict(int)})
    seen: set[int] = set()
    mapped = sorted(conn.execute("SELECT ticker, cik FROM sec_cik_map"), key=lambda r: (r[0] not in sectors, r[0]))
    for ticker, cik in mapped:  # a CIK with several tickers is counted once, under a known-sector ticker
        if cik in seen or cik not in have:
            continue
        seen.add(cik)
        bucket = per_sector[sectors.get(ticker, "Unknown")]
        bucket["companies"] += 1
        for concept in have[cik]:
            bucket["concepts"][concept] += 1
    groups, restated = conn.execute(
        """SELECT COUNT(*), SUM(n > 1) FROM (
               SELECT COUNT(DISTINCT value) AS n FROM sec_fundamentals
               GROUP BY cik, tag, unit, period_start, period_end)"""
    ).fetchone()
    facts, first_year, last_filed = conn.execute(
        "SELECT COUNT(*), MIN(substr(period_end, 1, 4)), MAX(filed_date) FROM sec_fundamentals"
    ).fetchone()
    starts = [
        r[0] for r in conn.execute(
            "SELECT MIN(substr(filed_date, 1, 4)) FROM sec_fundamentals GROUP BY cik"
        )
    ]
    return {
        "companies": len(seen),
        "facts": facts,
        "earliest_period_year": first_year,
        "latest_filed_date": last_filed,
        "median_first_filing_year": sorted(starts)[len(starts) // 2] if starts else None,
        "period_groups": groups,
        "period_groups_with_restated_values": restated or 0,
        "restated_pct": round(100.0 * (restated or 0) / groups, 2) if groups else 0.0,
        "sectors": {
            s: {"companies": b["companies"], "concept_coverage_pct": {
                c: round(100.0 * b["concepts"].get(c, 0) / b["companies"], 1) for c in cfg.CONCEPTS}}
            for s, b in sorted(per_sector.items())
        },
    }
