"""YOLO pattern analysis: delta comparisons, lifecycle tracking, persistence."""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

from trader_koo.report.utils import _parse_iso_date, table_exists

LOG = logging.getLogger(__name__)


def _yolo_match_tolerance_days(timeframe: Any) -> int:
    tf = str(timeframe or "").strip().lower()
    return 35 if tf == "weekly" else 14


def _yolo_snapshot_matches(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if str(anchor.get("ticker") or "").upper() != str(candidate.get("ticker") or "").upper():
        return False
    if str(anchor.get("timeframe") or "").strip().lower() != str(candidate.get("timeframe") or "").strip().lower():
        return False
    if str(anchor.get("pattern") or "").strip() != str(candidate.get("pattern") or "").strip():
        return False

    tolerance = _yolo_match_tolerance_days(anchor.get("timeframe"))
    anchor_x0 = _parse_iso_date(anchor.get("x0_date"))
    anchor_x1 = _parse_iso_date(anchor.get("x1_date"))
    cand_x0 = _parse_iso_date(candidate.get("x0_date"))
    cand_x1 = _parse_iso_date(candidate.get("x1_date"))

    x0_match = (
        anchor_x0 is not None
        and cand_x0 is not None
        and abs((anchor_x0 - cand_x0).days) <= tolerance
    )
    x1_match = (
        anchor_x1 is not None
        and cand_x1 is not None
        and abs((anchor_x1 - cand_x1).days) <= tolerance
    )
    return x0_match or x1_match


def _yolo_seen_streak(seen_asofs: set[str], asof_dates_desc: list[str], latest_asof: str | None = None) -> int:
    if not seen_asofs or not asof_dates_desc:
        return 0
    target_start = str(latest_asof or asof_dates_desc[0] or "")
    started = False
    streak = 0
    for asof in asof_dates_desc:
        if not started:
            if asof != target_start:
                continue
            started = True
        if asof in seen_asofs:
            streak += 1
        else:
            break
    return streak


def _summarize_yolo_lifecycle(
    anchor: dict[str, Any],
    history_rows: list[dict[str, Any]],
    asof_dates_desc: list[str],
) -> dict[str, Any]:
    current_asof = str(anchor.get("as_of_date") or "").strip() or None
    seen_asofs: set[str] = set()
    for row in history_rows:
        asof = str(row.get("as_of_date") or "").strip()
        if not asof:
            continue
        if _yolo_snapshot_matches(anchor, row):
            seen_asofs.add(asof)

    if not seen_asofs:
        return {
            "first_seen_asof": current_asof,
            "last_seen_asof": current_asof,
            "snapshots_seen": 1 if current_asof else 0,
            "current_streak": 1 if current_asof else 0,
            "first_seen_days_ago": 0 if current_asof else None,
        }

    seen_sorted = sorted(seen_asofs)
    first_seen = seen_sorted[0]
    last_seen = seen_sorted[-1]
    first_dt = _parse_iso_date(first_seen)
    current_dt = _parse_iso_date(current_asof)
    first_seen_days_ago = None
    if first_dt is not None and current_dt is not None:
        first_seen_days_ago = max(0, (current_dt - first_dt).days)

    return {
        "first_seen_asof": first_seen,
        "last_seen_asof": last_seen,
        "snapshots_seen": len(seen_asofs),
        "current_streak": _yolo_seen_streak(seen_asofs, asof_dates_desc, current_asof),
        "first_seen_days_ago": first_seen_days_ago,
    }


def fetch_yolo_delta(
    conn: sqlite3.Connection,
    timeframe: str | None = None,
    x0_tolerance_days: int = 14,
) -> dict[str, Any]:
    """Compare YOLO detections between latest two as_of dates (optionally per timeframe)."""
    tf = str(timeframe or "").strip().lower()
    if tf not in {"daily", "weekly"}:
        tf = ""
    delta: dict[str, Any] = {
        "timeframe": tf or "all",
        "today_asof": None,
        "prev_asof": None,
        "new_count": 0,
        "lost_count": 0,
        "new_patterns": [],
        "lost_patterns": [],
    }
    if not table_exists(conn, "yolo_patterns"):
        return delta

    if tf:
        asof_rows = conn.execute(
            """
            SELECT DISTINCT as_of_date
            FROM yolo_patterns
            WHERE timeframe = ?
            ORDER BY as_of_date DESC
            LIMIT 2
            """,
            (tf,),
        ).fetchall()
    else:
        asof_rows = conn.execute(
            """
            SELECT DISTINCT as_of_date
            FROM yolo_patterns
            ORDER BY as_of_date DESC
            LIMIT 2
            """
        ).fetchall()

    if len(asof_rows) < 2:
        if asof_rows:
            delta["today_asof"] = asof_rows[0][0]
        return delta

    today_asof = asof_rows[0][0]
    prev_asof = asof_rows[1][0]
    delta["today_asof"] = today_asof
    delta["prev_asof"] = prev_asof

    def _fetch_set(asof: str) -> list[dict[str, Any]]:
        if tf:
            rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ? AND timeframe = ?
                ORDER BY confidence DESC
                """,
                (asof, tf),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT ticker, timeframe, pattern,
                       CAST(confidence AS REAL) AS confidence,
                       x0_date, x1_date
                FROM yolo_patterns
                WHERE as_of_date = ?
                ORDER BY confidence DESC
                """,
                (asof,),
            ).fetchall()
        return [
            {
                "ticker": r[0],
                "timeframe": r[1],
                "pattern": r[2],
                "confidence": round(float(r[3]), 3),
                "x0_date": r[4],
                "x1_date": r[5],
            }
            for r in rows
        ]

    today_set = _fetch_set(today_asof)
    prev_set = _fetch_set(prev_asof)

    def _match_any(target: dict[str, Any], pool: list[dict[str, Any]]) -> bool:
        for candidate in pool:
            if (
                str(target.get("ticker") or "").upper() == str(candidate.get("ticker") or "").upper()
                and str(target.get("timeframe") or "").strip().lower() == str(candidate.get("timeframe") or "").strip().lower()
                and str(target.get("pattern") or "").strip() == str(candidate.get("pattern") or "").strip()
            ):
                t_x0 = _parse_iso_date(target.get("x0_date"))
                c_x0 = _parse_iso_date(candidate.get("x0_date"))
                t_x1 = _parse_iso_date(target.get("x1_date"))
                c_x1 = _parse_iso_date(candidate.get("x1_date"))
                tol = x0_tolerance_days
                x0_ok = t_x0 is not None and c_x0 is not None and abs((t_x0 - c_x0).days) <= tol
                x1_ok = t_x1 is not None and c_x1 is not None and abs((t_x1 - c_x1).days) <= tol
                if x0_ok or x1_ok:
                    return True
        return False

    new_patterns = [p for p in today_set if not _match_any(p, prev_set)]
    lost_patterns = [p for p in prev_set if not _match_any(p, today_set)]

    delta["new_count"] = len(new_patterns)
    delta["lost_count"] = len(lost_patterns)
    delta["new_patterns"] = new_patterns
    delta["lost_patterns"] = lost_patterns
    return delta


def fetch_yolo_pattern_persistence(
    conn: sqlite3.Connection,
    timeframe: str = "daily",
    lookback_asof: int = 20,
    top_n: int = 25,
) -> dict[str, Any]:
    """Measure how long each (ticker, pattern) has been consistently detected."""
    tf = str(timeframe or "daily").strip().lower()
    if tf not in {"daily", "weekly"}:
        tf = "daily"
    result: dict[str, Any] = {
        "timeframe": tf,
        "lookback_asof": 0,
        "latest_asof": None,
        "rows": [],
    }
    if not table_exists(conn, "yolo_patterns"):
        return result

    asof_rows = conn.execute(
        """
        SELECT DISTINCT as_of_date
        FROM yolo_patterns
        WHERE timeframe = ?
        ORDER BY as_of_date DESC
        LIMIT ?
        """,
        (tf, int(max(1, lookback_asof))),
    ).fetchall()
    asof_dates = [str(r[0]) for r in asof_rows if r[0]]
    if not asof_dates:
        return result
    latest_asof = asof_dates[0]
    result["latest_asof"] = latest_asof
    result["lookback_asof"] = len(asof_dates)

    # Fetch all detections within the lookback window.
    placeholders = ",".join("?" for _ in asof_dates)
    history_rows = conn.execute(
        f"""
        SELECT ticker, timeframe, pattern,
               CAST(confidence AS REAL) AS confidence,
               x0_date, x1_date, as_of_date
        FROM yolo_patterns
        WHERE timeframe = ? AND as_of_date IN ({placeholders})
        ORDER BY as_of_date DESC, confidence DESC
        """,
        [tf] + asof_dates,
    ).fetchall()
    history = [
        {
            "ticker": str(r[0] or ""),
            "timeframe": str(r[1] or ""),
            "pattern": str(r[2] or ""),
            "confidence": float(r[3] or 0.0),
            "x0_date": r[4],
            "x1_date": r[5],
            "as_of_date": str(r[6] or ""),
        }
        for r in history_rows
    ]

    # Anchor on latest_asof patterns.
    latest_patterns = [h for h in history if h["as_of_date"] == latest_asof]
    out_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for anchor in latest_patterns:
        key = (str(anchor.get("ticker") or "").upper(), str(anchor.get("pattern") or "").strip())
        if key in seen_keys:
            continue
        seen_keys.add(key)

        lifecycle = _summarize_yolo_lifecycle(anchor, history, asof_dates)
        seen_in = int(lifecycle.get("snapshots_seen") or 0)
        streak = int(lifecycle.get("current_streak") or 0)

        # Average confidence across matched snapshots.
        matched_confs = [
            float(h.get("confidence") or 0.0)
            for h in history
            if _yolo_snapshot_matches(anchor, h)
        ]
        avg_conf = round(sum(matched_confs) / len(matched_confs), 3) if matched_confs else None

        coverage = round((seen_in / max(1, len(asof_dates))) * 100.0, 1)
        out_rows.append(
            {
                "ticker": anchor.get("ticker"),
                "pattern": anchor.get("pattern"),
                "timeframe": tf,
                "latest_confidence": round(float(anchor.get("confidence") or 0.0), 3),
                "avg_confidence_window": avg_conf,
                "streak": streak,
                "seen_in_lookback": seen_in,
                "coverage_pct": coverage,
                "first_seen_asof": lifecycle.get("first_seen_asof"),
                "last_seen_asof": lifecycle.get("last_seen_asof"),
                "first_seen_days_ago": lifecycle.get("first_seen_days_ago"),
            }
        )

    out_rows.sort(
        key=lambda r: (
            int(r.get("streak") or 0),
            float(r.get("coverage_pct") or 0.0),
            float(r.get("latest_confidence") or 0.0),
        ),
        reverse=True,
    )
    result["rows"] = out_rows[: max(1, int(top_n))]
    return {
        **result,
        "lookback_asof": len(asof_dates),
        "rows": out_rows[: max(1, int(top_n))],
    }
