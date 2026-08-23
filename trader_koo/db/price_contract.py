"""Read the persisted price-basis contract before research uses a series."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

VALID_PRICE_BASES = {"split_adjusted_price_only", "total_return"}


def _sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def ensure_price_series_revision_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS price_series_revisions (
            ticker TEXT PRIMARY KEY,
            managed_start TEXT NOT NULL,
            managed_end TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            adjustment_basis TEXT NOT NULL,
            adjustment_version TEXT NOT NULL,
            price_sha256 TEXT NOT NULL,
            action_sha256 TEXT NOT NULL,
            evidence_sha256 TEXT NOT NULL,
            revision_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            fetch_timestamp TEXT NOT NULL
        )"""
    )


def _series_material(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    if not _table_exists(conn, "price_daily"):
        return None
    prices = conn.execute(
        """SELECT date,open,high,low,close,volume,adjustment_basis,
                  adjustment_version,basis_status,unresolved_reason
           FROM price_daily WHERE ticker=? ORDER BY date""",
        (ticker,),
    ).fetchall()
    if not prices:
        return None
    actions = []
    if _table_exists(conn, "price_corporate_actions"):
        actions = conn.execute(
            """SELECT action_date,action_type,provider,value,applied_to_prices,
                      adjustment_version,fetch_timestamp,evidence_json
               FROM price_corporate_actions WHERE ticker=?
               ORDER BY action_date,action_type,provider""",
            (ticker,),
        ).fetchall()
    basis_pairs = {(row[6], row[7]) for row in prices}
    basis, version = next(iter(basis_pairs)) if len(basis_pairs) == 1 else ("unknown", "unknown")
    return {
        "managed_start": str(prices[0][0]),
        "managed_end": str(prices[-1][0]),
        "row_count": len(prices),
        "basis": str(basis or "unknown"),
        "version": str(version or "unknown"),
        "rows_verified": all(row[8] == "verified" and not row[9] for row in prices),
        "price_sha256": _sha256([list(row) for row in prices]),
        "action_sha256": _sha256([list(row) for row in actions]),
    }


def record_price_series_revision(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    evidence: dict[str, Any],
    fetch_timestamp: str,
) -> dict[str, Any]:
    """Seal the exact persisted price/action rows plus the provider audit."""
    ensure_price_series_revision_schema(conn)
    material = _series_material(conn, ticker)
    if material is None:
        raise ValueError(f"cannot seal missing price series for {ticker}")
    evidence_sha = _sha256(evidence)
    verified = (
        material["rows_verified"]
        and valid_price_provenance(material["basis"], material["version"])
        and evidence.get("vendor_action_ledger_checked") is True
    )
    revision = _sha256({
        "ticker": ticker,
        "managed_start": material["managed_start"],
        "managed_end": material["managed_end"],
        "row_count": material["row_count"],
        "basis": material["basis"],
        "version": material["version"],
        "price_sha256": material["price_sha256"],
        "action_sha256": material["action_sha256"],
        "evidence_sha256": evidence_sha,
    })
    previous = conn.execute(
        "SELECT managed_start,managed_end,action_sha256,revision_sha256 "
        "FROM price_series_revisions WHERE ticker=?",
        (ticker,),
    ).fetchone()
    materially_reseeded = previous is not None and previous[3] != revision and (
        previous[0] != material["managed_start"]
        or previous[1] == material["managed_end"]
        or previous[2] != material["action_sha256"]
    )
    if materially_reseeded and _table_exists(conn, "yolo_patterns"):
        conn.execute("DELETE FROM yolo_patterns WHERE ticker=?", (ticker,))
    conn.execute(
        """INSERT INTO price_series_revisions (
               ticker,managed_start,managed_end,row_count,adjustment_basis,
               adjustment_version,price_sha256,action_sha256,evidence_sha256,
               revision_sha256,status,evidence_json,fetch_timestamp
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker) DO UPDATE SET
               managed_start=excluded.managed_start,
               managed_end=excluded.managed_end,
               row_count=excluded.row_count,
               adjustment_basis=excluded.adjustment_basis,
               adjustment_version=excluded.adjustment_version,
               price_sha256=excluded.price_sha256,
               action_sha256=excluded.action_sha256,
               evidence_sha256=excluded.evidence_sha256,
               revision_sha256=excluded.revision_sha256,
               status=excluded.status,
               evidence_json=excluded.evidence_json,
               fetch_timestamp=excluded.fetch_timestamp""",
        (
            ticker, material["managed_start"], material["managed_end"],
            material["row_count"], material["basis"], material["version"],
            material["price_sha256"], material["action_sha256"], evidence_sha,
            revision, "verified" if verified else "unresolved",
            json.dumps(evidence, sort_keys=True, separators=(",", ":")), fetch_timestamp,
        ),
    )
    return {**material, "evidence_sha256": evidence_sha, "revision": revision, "eligible": verified}


def _verified_revision(conn: sqlite3.Connection, ticker: str) -> dict[str, Any] | None:
    if not _table_exists(conn, "price_series_revisions"):
        return None
    row = conn.execute(
        """SELECT managed_start,managed_end,row_count,adjustment_basis,
                  adjustment_version,price_sha256,action_sha256,evidence_sha256,
                  revision_sha256,status,evidence_json
           FROM price_series_revisions WHERE ticker=?""",
        (ticker,),
    ).fetchone()
    material = _series_material(conn, ticker)
    if row is None or material is None:
        return None
    expected_revision = _sha256({
        "ticker": ticker,
        "managed_start": material["managed_start"],
        "managed_end": material["managed_end"],
        "row_count": material["row_count"],
        "basis": material["basis"],
        "version": material["version"],
        "price_sha256": material["price_sha256"],
        "action_sha256": material["action_sha256"],
        "evidence_sha256": row[7],
    })
    matches = (
        tuple(row[:7]) == (
            material["managed_start"], material["managed_end"], material["row_count"],
            material["basis"], material["version"], material["price_sha256"],
            material["action_sha256"],
        )
        and row[8] == expected_revision
        and row[9] == "verified"
        and material["rows_verified"]
    )
    try:
        evidence = json.loads(str(row[10]))
    except (TypeError, ValueError):
        evidence = None
    if not isinstance(evidence, dict) or _sha256(evidence) != row[7]:
        matches = False
    return {
        "eligible": matches,
        "basis": material["basis"],
        "version": material["version"],
        "managed_start": material["managed_start"],
        "managed_end": material["managed_end"],
        "row_count": material["row_count"],
        "price_sha256": material["price_sha256"],
        "action_sha256": material["action_sha256"],
        "evidence_sha256": str(row[7]),
        "revision": expected_revision,
    }


def valid_price_provenance(basis: object, version: object) -> bool:
    return (
        str(basis or "").strip() in VALID_PRICE_BASES
        and str(version or "").strip().lower() not in {"", "unknown"}
    )


def research_eligible_tickers(conn: sqlite3.Connection) -> set[str]:
    """Return only series whose current rows match their sealed revision."""
    if not _table_exists(conn, "price_daily"):
        return set()
    tickers = [str(row[0]) for row in conn.execute("SELECT DISTINCT ticker FROM price_daily")]
    return {ticker for ticker in tickers if (_verified_revision(conn, ticker) or {}).get("eligible")}


def research_price_contract(
    conn: sqlite3.Connection,
    tickers: list[str] | None = None,
) -> dict[str, Any]:
    if not _table_exists(conn, "price_daily") or not _table_exists(conn, "price_series_revisions"):
        return {
            "eligible": False,
            "basis": "unknown",
            "version": "unknown",
            "status": "unverified",
            "reason": "price_series_revision_unavailable",
        }
    requested = list(dict.fromkeys(tickers or [
        str(row[0]) for row in conn.execute("SELECT DISTINCT ticker FROM price_daily ORDER BY ticker")
    ]))
    contracts = {ticker: _verified_revision(conn, ticker) for ticker in requested}
    missing_tickers = sorted(ticker for ticker, contract in contracts.items() if contract is None)
    present = [contract for contract in contracts.values() if contract is not None]
    bases = {(contract["basis"], contract["version"]) for contract in present}
    basis, version = next(iter(bases)) if len(bases) == 1 else ("unknown", "unknown")
    verified = bool(present) and not missing_tickers and len(bases) == 1 and all(
        contract["eligible"] for contract in present
    )
    revision = _sha256({ticker: contract["revision"] for ticker, contract in contracts.items()}) if verified else None
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
        "revision": revision,
        "ticker_contracts": contracts,
        "managed_window": {
            "start": min((contract["managed_start"] for contract in present), default=None),
            "end": max((contract["managed_end"] for contract in present), default=None),
        },
    }
