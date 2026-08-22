"""Read the persisted price-basis contract before research uses a series."""
from __future__ import annotations

import sqlite3
from typing import Any


def research_price_contract(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(price_daily)").fetchall()}
    required = {"adjustment_basis", "adjustment_version", "basis_status"}
    if not required.issubset(columns):
        return {
            "eligible": False,
            "basis": "unknown",
            "version": "unknown",
            "status": "unverified",
            "reason": "price_basis_schema_not_migrated",
        }

    params: tuple[str, ...] = tuple(dict.fromkeys(tickers or []))
    where = ""
    if params:
        where = f"WHERE ticker IN ({','.join('?' for _ in params)})"
    rows = conn.execute(
        f"""
        SELECT adjustment_basis, adjustment_version, basis_status
        FROM price_daily {where}
        GROUP BY adjustment_basis, adjustment_version, basis_status
        """,
        params,
    ).fetchall()
    bases = {(row[0], row[1]) for row in rows if row[0] and row[1]}
    verified = bool(rows) and len(bases) == 1 and all(row[2] == "verified" for row in rows)
    basis, version = next(iter(bases)) if len(bases) == 1 else ("unknown", "unknown")
    return {
        "eligible": verified,
        "basis": basis,
        "version": version,
        "status": "verified" if verified else "unresolved",
        "reason": None if verified else "mixed_or_unresolved_price_basis",
        "distributions_included": basis == "total_return",
    }
