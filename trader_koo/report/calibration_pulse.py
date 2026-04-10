"""Calibration pulse — runs every 3 trading sessions to keep setup scoring honest.

Reads outcomes from two sources and writes per-(family, direction) adjustments
to the ``calibration_state`` table:

  1. ``setup_call_evaluations`` — all 40 setups per report over 90 days (broad)
  2. ``paper_trades`` — critic-filtered, fully closed trades (gold standard)

These are combined with paper_trades weighted 2× to reflect that they represent
real capital decisions, not just directional calls. The pulse writes:

  - ``score_adjustment``: added to raw confluence score at report generation time
    (range: SCORE_ADJ_MIN to SCORE_ADJ_MAX)
  - ``block_new_entries``: critic rejects setup entirely when True
    (triggered by sustained negative expectancy over a sufficient sample)

Scheduling: Mon / Wed / Fri at 23:15 UTC — 75 min after the nightly report.
Manual trigger: POST /api/admin/calibration/run-pulse

The scorer and critic both read calibration_state at runtime. If the table does
not yet exist (fresh deploy, first report) they fail open and use defaults.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from typing import Any

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — document every constant.
# To change a threshold: update the constant + update tests/test_calibration_pulse.py
# + PR body must include the before/after impact on family blocks.
# ---------------------------------------------------------------------------

EVAL_WINDOW_DAYS: int = 90         # look back N days in setup_call_evaluations
PAPER_LOOKBACK: int = 30           # last N closed paper trades
PAPER_WEIGHT: float = 2.0          # paper_trades count as this many eval samples

MIN_EVAL_SAMPLE: int = 15          # min setup_call_evaluations before adjusting
MIN_PAPER_SAMPLE: int = 5          # min closed paper trades before adjusting
MIN_COMBINED_FOR_BLOCK: int = 20   # min combined samples to trigger block_new_entries

BLOCK_EXPECTANCY_THRESHOLD: float = -1.5   # block if expectancy < this (%)
BLOCK_RESTORE_EXPECTANCY: float = 0.0      # lift block once expectancy returns >= 0%
BLOCK_HIT_RATE_THRESHOLD: float = 38.0    # also block if hit_rate < this (%)

SCORE_ADJ_MAX: float = 5.0         # cap positive adjustments (avoid overconfident upward bias)
SCORE_ADJ_MIN: float = -15.0       # max demotion (stronger than the old ±10 range)

# Tiers of score adjustment by expectancy
_TIER_THRESHOLDS: list[tuple[float, float]] = [
    # (expectancy_pct threshold, adjustment)  ordered from most negative to most positive
    (-2.0, -15.0),
    (-1.5, -10.0),
    (-1.0, -7.0),
    (-0.5, -4.0),
    (0.0,  -1.0),
    (0.5,   2.0),
    (1.0,   4.0),
    (2.0,   5.0),
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_calibration_schema(conn: sqlite3.Connection) -> None:
    """Create calibration_state table (idempotent)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family TEXT NOT NULL,
            direction TEXT NOT NULL,
            score_adjustment REAL NOT NULL DEFAULT 0.0,
            block_new_entries INTEGER NOT NULL DEFAULT 0,
            hit_rate_pct REAL,
            expectancy_pct REAL,
            combined_sample_count INTEGER,
            eval_sample_count INTEGER,
            paper_sample_count INTEGER,
            last_updated TEXT NOT NULL,
            last_updated_trigger TEXT,
            notes TEXT,
            UNIQUE(family, direction)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_calibration_state_family "
        "ON calibration_state(family, direction)"
    )
    conn.commit()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
    )


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _eval_stats(
    conn: sqlite3.Connection,
    *,
    window_days: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read setup_call_evaluations outcomes for the last N days."""
    if not _table_exists(conn, "setup_call_evaluations"):
        return {}
    cutoff = (
        dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=window_days)
    ).isoformat()
    rows = conn.execute(
        """
        SELECT setup_family, call_direction, direction_hit, signed_return_pct
        FROM setup_call_evaluations
        WHERE status = 'scored'
          AND asof_date >= ?
          AND setup_family IS NOT NULL
          AND call_direction IN ('long', 'short')
        """,
        (cutoff,),
    ).fetchall()

    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row[0]).lower().replace(" ", "_"), str(row[1]).lower())
        ret = float(row[3]) if row[3] is not None else 0.0
        buckets.setdefault(key, []).append(ret)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, returns in buckets.items():
        n = len(returns)
        wins = sum(1 for r in returns if r > 0)
        out[key] = {
            "sample": n,
            "hit_rate_pct": wins / n * 100.0 if n else 0.0,
            "expectancy_pct": sum(returns) / n if n else 0.0,
        }
    return out


def _paper_stats(
    conn: sqlite3.Connection,
    *,
    lookback: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read closed paper trades for the last N trades per family."""
    if not _table_exists(conn, "paper_trades"):
        return {}
    rows = conn.execute(
        """
        SELECT setup_family, direction, pnl_pct
        FROM paper_trades
        WHERE status != 'open'
          AND pnl_pct IS NOT NULL
          AND setup_family IS NOT NULL
          AND direction IN ('long', 'short')
        ORDER BY exit_date DESC
        LIMIT ?
        """,
        (lookback * 10,),  # over-fetch then slice per family
    ).fetchall()

    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (str(row[0]).lower().replace(" ", "_"), str(row[1]).lower())
        buckets.setdefault(key, []).append(float(row[2]))

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, returns in buckets.items():
        # Cap per-family lookback to avoid distant history dominating
        recent = returns[:lookback]
        n = len(recent)
        wins = sum(1 for r in recent if r > 0)
        out[key] = {
            "sample": n,
            "hit_rate_pct": wins / n * 100.0 if n else 0.0,
            "expectancy_pct": sum(recent) / n if n else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# Combining + scoring
# ---------------------------------------------------------------------------

def _combined_expectancy(
    eval_stat: dict[str, Any] | None,
    paper_stat: dict[str, Any] | None,
    *,
    paper_weight: float,
) -> dict[str, Any]:
    """Weighted average of eval and paper stats.

    Paper trades count as ``paper_weight`` eval samples each.
    Returns combined hit_rate, expectancy, and sample counts.
    """
    eval_n = int((eval_stat or {}).get("sample") or 0)
    paper_n = int((paper_stat or {}).get("sample") or 0)

    # Weighted returns list
    eval_returns: list[float] = []
    paper_returns: list[float] = []

    if eval_stat and eval_n > 0:
        # Reconstruct approximate returns from hit_rate + expectancy
        # (exact values not available at this aggregation level — use expectancy as proxy)
        eval_returns = [float(eval_stat["expectancy_pct"])] * eval_n

    if paper_stat and paper_n > 0:
        # Paper trades weighted more heavily
        paper_returns = [float(paper_stat["expectancy_pct"])] * int(paper_n * paper_weight)

    all_returns = eval_returns + paper_returns
    combined_n = len(all_returns)

    if combined_n == 0:
        return {
            "combined_sample": 0,
            "eval_sample": eval_n,
            "paper_sample": paper_n,
            "hit_rate_pct": None,
            "expectancy_pct": None,
        }

    # For hit_rate: weighted average of the two hit rates
    eval_hr = float((eval_stat or {}).get("hit_rate_pct") or 0.0)
    paper_hr = float((paper_stat or {}).get("hit_rate_pct") or 0.0)
    eval_weight = eval_n
    paper_effective = int(paper_n * paper_weight)
    total_weight = eval_weight + paper_effective
    if total_weight > 0:
        combined_hr = (eval_hr * eval_weight + paper_hr * paper_effective) / total_weight
    else:
        combined_hr = 0.0

    combined_exp = sum(all_returns) / combined_n

    return {
        "combined_sample": eval_n + paper_n,  # actual trades (not weighted)
        "eval_sample": eval_n,
        "paper_sample": paper_n,
        "hit_rate_pct": round(combined_hr, 1),
        "expectancy_pct": round(combined_exp, 2),
    }


def _compute_score_adjustment(combined: dict[str, Any]) -> float:
    """Map combined expectancy to a score adjustment."""
    exp = combined.get("expectancy_pct")
    if exp is None:
        return 0.0
    for threshold, adj in _TIER_THRESHOLDS:
        if exp < threshold:
            return adj
    return SCORE_ADJ_MAX  # very positive expectancy


def _compute_block(combined: dict[str, Any]) -> bool:
    """True if the family should be blocked entirely."""
    n = int(combined.get("combined_sample") or 0)
    if n < MIN_COMBINED_FOR_BLOCK:
        return False
    exp = combined.get("expectancy_pct")
    hr = combined.get("hit_rate_pct")
    if exp is None or hr is None:
        return False
    return exp < BLOCK_EXPECTANCY_THRESHOLD and hr < BLOCK_HIT_RATE_THRESHOLD


def _make_notes(combined: dict[str, Any], adj: float, block: bool) -> str:
    exp = combined.get("expectancy_pct")
    hr = combined.get("hit_rate_pct")
    n_combined = combined.get("combined_sample", 0)
    n_paper = combined.get("paper_sample", 0)
    n_eval = combined.get("eval_sample", 0)
    parts = [f"{n_combined} combined ({n_paper} paper + {n_eval} eval)"]
    if hr is not None:
        parts.append(f"hit={hr:.0f}%")
    if exp is not None:
        parts.append(f"exp={exp:.2f}%")
    parts.append(f"adj={adj:+.1f}pts")
    if block:
        parts.append("BLOCKED")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_calibration_pulse(
    conn: sqlite3.Connection,
    *,
    trigger: str = "scheduled",
) -> dict[str, Any]:
    """Recompute calibration_state for all known (family, direction) pairs.

    Safe to call multiple times — uses UPSERT. Returns a summary dict
    suitable for Telegram notification.
    """
    ensure_calibration_schema(conn)

    eval_stats = _eval_stats(conn, window_days=EVAL_WINDOW_DAYS)
    paper_stats = _paper_stats(conn, lookback=PAPER_LOOKBACK)

    # Union of all (family, direction) pairs seen in either source
    all_keys: set[tuple[str, str]] = set(eval_stats) | set(paper_stats)

    now_ts = dt.datetime.now(dt.timezone.utc).isoformat()
    changes: list[dict[str, Any]] = []
    blocks_added: list[str] = []
    blocks_lifted: list[str] = []

    # Read existing state so we can detect changes
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for row in conn.execute(
            "SELECT family, direction, score_adjustment, block_new_entries FROM calibration_state"
        ).fetchall():
            existing[(str(row[0]), str(row[1]))] = {
                "score_adjustment": float(row[2]),
                "block_new_entries": bool(row[3]),
            }
    except Exception:
        pass

    for family, direction in sorted(all_keys):
        eval_s = eval_stats.get((family, direction))
        paper_s = paper_stats.get((family, direction))

        # Skip if we don't have enough data in either source
        eval_ok = eval_s is not None and int(eval_s.get("sample") or 0) >= MIN_EVAL_SAMPLE
        paper_ok = paper_s is not None and int(paper_s.get("sample") or 0) >= MIN_PAPER_SAMPLE
        if not eval_ok and not paper_ok:
            continue

        combined = _combined_expectancy(eval_s, paper_s, paper_weight=PAPER_WEIGHT)
        adj = _compute_score_adjustment(combined)
        adj = max(SCORE_ADJ_MIN, min(SCORE_ADJ_MAX, adj))
        block = _compute_block(combined)
        notes = _make_notes(combined, adj, block)

        # Detect changes vs existing state
        prev = existing.get((family, direction))
        key_str = f"{family}:{direction}"
        if prev:
            prev_block = prev["block_new_entries"]
            prev_adj = prev["score_adjustment"]
            if block and not prev_block:
                blocks_added.append(key_str)
            elif not block and prev_block:
                blocks_lifted.append(key_str)
            if abs(adj - prev_adj) >= 1.0 or block != prev_block:
                changes.append({
                    "key": key_str,
                    "old_adj": prev_adj,
                    "new_adj": adj,
                    "old_block": prev_block,
                    "new_block": block,
                    "expectancy_pct": combined.get("expectancy_pct"),
                    "hit_rate_pct": combined.get("hit_rate_pct"),
                    "combined_sample": combined.get("combined_sample"),
                })
        else:
            changes.append({
                "key": key_str,
                "old_adj": None,
                "new_adj": adj,
                "old_block": None,
                "new_block": block,
                "expectancy_pct": combined.get("expectancy_pct"),
                "hit_rate_pct": combined.get("hit_rate_pct"),
                "combined_sample": combined.get("combined_sample"),
            })

        conn.execute(
            """
            INSERT INTO calibration_state
                (family, direction, score_adjustment, block_new_entries,
                 hit_rate_pct, expectancy_pct, combined_sample_count,
                 eval_sample_count, paper_sample_count,
                 last_updated, last_updated_trigger, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(family, direction) DO UPDATE SET
                score_adjustment       = excluded.score_adjustment,
                block_new_entries      = excluded.block_new_entries,
                hit_rate_pct           = excluded.hit_rate_pct,
                expectancy_pct         = excluded.expectancy_pct,
                combined_sample_count  = excluded.combined_sample_count,
                eval_sample_count      = excluded.eval_sample_count,
                paper_sample_count     = excluded.paper_sample_count,
                last_updated           = excluded.last_updated,
                last_updated_trigger   = excluded.last_updated_trigger,
                notes                  = excluded.notes
            """,
            (
                family,
                direction,
                round(adj, 1),
                1 if block else 0,
                combined.get("hit_rate_pct"),
                combined.get("expectancy_pct"),
                combined.get("combined_sample"),
                combined.get("eval_sample"),
                combined.get("paper_sample"),
                now_ts,
                trigger,
                notes,
            ),
        )

    conn.commit()

    summary = {
        "ok": True,
        "trigger": trigger,
        "ts": now_ts,
        "families_updated": len(all_keys),
        "changes": changes,
        "blocks_added": blocks_added,
        "blocks_lifted": blocks_lifted,
    }
    LOG.info(
        "Calibration pulse complete: families=%d changes=%d blocks_added=%s blocks_lifted=%s",
        len(all_keys),
        len(changes),
        blocks_added,
        blocks_lifted,
    )
    return summary


def load_calibration_state(
    conn: sqlite3.Connection,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load calibration_state into a lookup dict keyed by (family, direction).

    Returns empty dict if the table doesn't exist yet (fail open).
    """
    if not _table_exists(conn, "calibration_state"):
        return {}
    try:
        rows = conn.execute(
            """
            SELECT family, direction, score_adjustment, block_new_entries,
                   hit_rate_pct, expectancy_pct, combined_sample_count, last_updated
            FROM calibration_state
            """
        ).fetchall()
    except Exception:
        return {}
    return {
        (str(r[0]), str(r[1])): {
            "score_adjustment": float(r[2]),
            "block_new_entries": bool(r[3]),
            "hit_rate_pct": r[4],
            "expectancy_pct": r[5],
            "combined_sample_count": r[6],
            "last_updated": r[7],
        }
        for r in rows
    }


def build_telegram_message(summary: dict[str, Any]) -> str:
    """Format a calibration pulse summary for Telegram."""
    ts = str(summary.get("ts") or "")[:19].replace("T", " ")
    trigger = summary.get("trigger", "scheduled")
    n_updated = summary.get("families_updated", 0)
    changes = summary.get("changes") or []
    blocks_added = summary.get("blocks_added") or []
    blocks_lifted = summary.get("blocks_lifted") or []

    lines = [f"*Calibration Pulse* ({trigger}) — {ts} UTC"]
    lines.append(f"Families evaluated: {n_updated}")

    if blocks_added:
        lines.append("\n*Blocked (negative edge):*")
        for key in blocks_added:
            change = next((c for c in changes if c["key"] == key), {})
            exp = change.get("expectancy_pct")
            hr = change.get("hit_rate_pct")
            n = change.get("combined_sample")
            lines.append(
                f"  \u274c {key}"
                + (f" exp={exp:.1f}%" if exp is not None else "")
                + (f" hit={hr:.0f}%" if hr is not None else "")
                + (f" ({n} samples)" if n else "")
            )

    if blocks_lifted:
        lines.append("\n*Restored (edge recovered):*")
        for key in blocks_lifted:
            lines.append(f"  \u2705 {key}")

    # Show largest score adjustments that aren't blocks
    score_changes = [c for c in changes if c["key"] not in blocks_added and c["key"] not in blocks_lifted]
    score_changes.sort(key=lambda c: abs((c.get("new_adj") or 0) - (c.get("old_adj") or 0)), reverse=True)
    if score_changes:
        lines.append("\n*Score adjustments:*")
        for c in score_changes[:5]:
            old = c.get("old_adj")
            new = c.get("new_adj")
            exp = c.get("expectancy_pct")
            old_str = f"{old:+.1f}" if old is not None else "new"
            new_str = f"{new:+.1f}" if new is not None else "0"
            lines.append(
                f"  {c['key']}: {old_str} → {new_str}"
                + (f" (exp={exp:.1f}%)" if exp is not None else "")
            )

    if not blocks_added and not blocks_lifted and not score_changes:
        lines.append("No significant changes.")

    return "\n".join(lines)
