"""Read the persisted price-basis contract before research uses a series."""
from __future__ import annotations

import sqlite3
from typing import Any

VALID_PRICE_BASES = {"split_adjusted_price_only", "total_return"}


def valid_price_provenance(basis: object, version: object) -> bool:
    return (
        str(basis or "").strip() in VALID_PRICE_BASES
        and str(version or "").strip().lower() not in {"", "unknown"}
    )


def research_eligible_tickers(conn: sqlite3.Connection) -> set[str]:
    """Return tickers whose every persisted row has one verified known basis."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(price_daily)")}
    if not {"adjustment_basis", "adjustment_version", "basis_status"}.issubset(columns):
        return set()
    rows = conn.execute(
        """
        SELECT ticker, adjustment_basis, adjustment_version
        FROM price_daily GROUP BY ticker
        HAVING COUNT(DISTINCT COALESCE(adjustment_basis, '') || '|' ||
               COALESCE(adjustment_version, '')) = 1
           AND MIN(COALESCE(basis_status, 'unverified')) = 'verified'
           AND MAX(COALESCE(basis_status, 'unverified')) = 'verified'
        """
    ).fetchall()
    return {
        str(row[0]) for row in rows if valid_price_provenance(row[1], row[2])
    }


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
    missing_tickers: list[str] = []
    if params:
        present = {
            str(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT ticker FROM price_daily WHERE ticker IN ({','.join('?' for _ in params)})",
                params,
            ).fetchall()
        }
        missing_tickers = sorted(set(params) - present)
    bases = {(row[0], row[1]) for row in rows if row[0] and row[1]}
    basis, version = next(iter(bases)) if len(bases) == 1 else ("unknown", "unknown")
    verified = (
        bool(rows)
        and not missing_tickers
        and len(bases) == 1
        and valid_price_provenance(basis, version)
        and all(valid_price_provenance(row[0], row[1]) for row in rows)
        and all(row[2] == "verified" for row in rows)
    )
    return {
        "eligible": verified,
        "basis": basis,
        "version": version,
        "status": "verified" if verified else "unresolved",
        "reason": (
            None
            if verified
            else "missing_requested_tickers"
            if missing_tickers
            else "mixed_or_unresolved_price_basis"
        ),
        "missing_tickers": missing_tickers,
        "distributions_included": basis == "total_return",
    }
