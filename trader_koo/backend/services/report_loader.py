"""Daily report file loading and lookup.

Handles locating, parsing, and extracting setup data from the
``daily_report_*.json`` files stored in the report directory.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any

from trader_koo.backend.services.market_data import parse_iso_utc

LOG = logging.getLogger("trader_koo.services.report_loader")

# Maximum age (hours) before report cache is considered stale.
# After market close (22:00 UTC pipeline) the report is fresh.
# By next market close (~26h later) it's stale.
_REPORT_CACHE_MAX_AGE_HOURS = 26


def is_report_fresh(payload: dict[str, Any] | None) -> bool:
    """Check if the report is recent enough to serve from cache.

    Returns True if report was generated within the last trading day
    (accounting for weekends — Friday's report is valid until Monday evening).
    """
    if not isinstance(payload, dict):
        return False
    generated_ts = parse_iso_utc(payload.get("generated_ts"))
    if generated_ts is None:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    age_hours = (now - generated_ts).total_seconds() / 3600
    # Friday 22:00 UTC → Monday 22:00 UTC = 72 hours
    # Allow up to 74 hours to cover weekends + holidays
    weekday = generated_ts.weekday()  # 0=Mon, 4=Fri
    max_age = 74 if weekday == 4 else _REPORT_CACHE_MAX_AGE_HOURS
    return age_hours <= max_age


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _load_json_file(path: Path) -> dict[str, Any] | None:
    """Read and parse a JSON file, returning None on any failure."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Failed to parse JSON file %s: %s", path.name, exc)
        return None


def _tail_text_file(
    path: Path,
    lines: int = 60,
    max_bytes: int = 64_000,
) -> list[str]:
    """Return the last *lines* lines from *path*, reading at most *max_bytes*."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception as exc:
        LOG.warning("Failed to tail file %s: %s", path.name, exc)
        return []


# ---------------------------------------------------------------------------
# Report discovery
# ---------------------------------------------------------------------------

def latest_daily_report_json(
    report_dir: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Find and parse the latest daily report JSON.

    Checks ``daily_report_latest.json`` first, then falls back to
    the newest dated report file.
    """
    latest = report_dir / "daily_report_latest.json"
    payload = _load_json_file(latest)
    if payload is not None:
        return latest, payload
    candidates = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )
    for p in candidates:
        payload = _load_json_file(p)
        if payload is not None:
            return p, payload
    return None, None


def report_json_for_generated_ts(
    report_dir: Path,
    generated_ts: str | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Locate a report matching *generated_ts*, falling back to the latest."""
    target = parse_iso_utc(generated_ts)
    if target is None:
        return latest_daily_report_json(report_dir)
    target = target.replace(microsecond=0)
    target_iso = target.isoformat().replace("+00:00", "Z")

    candidate_path = report_dir / f"daily_report_{target.strftime('%Y%m%dT%H%M%SZ')}.json"
    payload = _load_json_file(candidate_path)
    if payload is not None:
        return candidate_path, payload

    latest_path, latest_payload = latest_daily_report_json(report_dir)
    latest_generated = (
        parse_iso_utc((latest_payload or {}).get("generated_ts"))
        if isinstance(latest_payload, dict)
        else None
    )
    if (
        latest_payload is not None
        and latest_generated is not None
        and latest_generated.replace(microsecond=0) == target
    ):
        return latest_path, latest_payload

    candidates = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )
    for p in candidates[:120]:
        payload = _load_json_file(p)
        if not isinstance(payload, dict):
            continue
        row_ts = parse_iso_utc(payload.get("generated_ts"))
        if row_ts is None:
            continue
        if row_ts.replace(microsecond=0).isoformat().replace("+00:00", "Z") == target_iso:
            return p, payload
    return latest_daily_report_json(report_dir)


# ---------------------------------------------------------------------------
# Setup-row extraction
# ---------------------------------------------------------------------------

def _extract_report_setup_rows(
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Pull setup rows from every known key in the ``signals`` block."""
    if not isinstance(payload, dict):
        return []
    signals = payload.get("signals")
    if not isinstance(signals, dict):
        return []
    out: list[dict[str, Any]] = []
    for key in ("setup_quality_top", "setup_quality_all", "setup_quality", "watchlist_candidates"):
        rows = signals.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    out.append(row)
    return out


def _normalize_setup_row(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map alternative field names to canonical ones."""
    if not isinstance(raw, dict):
        return None
    row = dict(raw)
    if row.get("setup_tier") in (None, "") and row.get("tier") not in (None, ""):
        row["setup_tier"] = row.get("tier")
    if row.get("score") in (None, "") and row.get("setup_score") not in (None, ""):
        row["score"] = row.get("setup_score")
    if row.get("setup_family") in (None, "") and row.get("setup") not in (None, ""):
        row["setup_family"] = row.get("setup")
    if row.get("signal_bias") in (None, "") and row.get("bias") not in (None, ""):
        row["signal_bias"] = row.get("bias")
    if row.get("actionability") in (None, "") and row.get("state") not in (None, ""):
        row["actionability"] = row.get("state")
    if row.get("observation") in (None, "") and row.get("what_it_is") not in (None, ""):
        row["observation"] = row.get("what_it_is")
    if row.get("action") in (None, "") and row.get("next_step") not in (None, ""):
        row["action"] = row.get("next_step")
    if row.get("risk_note") in (None, "") and row.get("risk") not in (None, ""):
        row["risk_note"] = row.get("risk")
    if row.get("technical_read") in (None, "") and row.get("technical_context") not in (None, ""):
        row["technical_read"] = row.get("technical_context")
    if row.get("yolo_signal_role") in (None, "") and row.get("yolo_role") not in (None, ""):
        row["yolo_signal_role"] = row.get("yolo_role")
    return row


def latest_report_setup_for_ticker(
    report_dir: Path,
    ticker: str,
    *,
    generated_ts: str | None = None,
) -> dict[str, Any] | None:
    """Look up the report-snapshot setup for *ticker*.

    Uses the ``setup_quality_lookup`` dict first (O(1)), then falls back
    to a linear scan of setup rows.
    """
    _, payload = report_json_for_generated_ts(report_dir, generated_ts)
    target = str(ticker or "").strip().upper()
    if isinstance(payload, dict):
        signals = payload.get("signals")
        if isinstance(signals, dict):
            lookup = signals.get("setup_quality_lookup")
            if isinstance(lookup, dict):
                row = lookup.get(target) or lookup.get(target.upper()) or lookup.get(target.lower())
                out = _normalize_setup_row(row if isinstance(row, dict) else None)
                if isinstance(out, dict):
                    out.setdefault("ticker", target)
                    return out
    for row in _extract_report_setup_rows(payload):
        if str(row.get("ticker") or "").strip().upper() == target:
            return _normalize_setup_row(dict(row))
    return None


def latest_report_hmm_for_ticker(
    report_dir: Path,
    ticker: str,
    *,
    generated_ts: str | None = None,
) -> dict[str, Any] | None:
    """Load pre-computed HMM regime for *ticker* from a fresh report snapshot."""
    _, payload = report_json_for_generated_ts(report_dir, generated_ts)
    if not is_report_fresh(payload):
        return None
    target = str(ticker or "").strip().upper()
    signals = payload.get("signals")
    if not isinstance(signals, dict):
        return None
    hmm_lookup = signals.get("hmm_regime_by_ticker")
    if not isinstance(hmm_lookup, dict):
        return None
    return hmm_lookup.get(target)


# ---------------------------------------------------------------------------
# Report history
# ---------------------------------------------------------------------------

def daily_report_history(
    report_dir: Path,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return metadata for the most recent *limit* report files."""
    out: list[dict[str, Any]] = []
    files = sorted(
        [p for p in report_dir.glob("daily_report_*.json") if p.name != "daily_report_latest.json"],
        key=lambda p: p.name,
        reverse=True,
    )[: max(1, limit)]
    for p in files:
        try:
            st = p.stat()
            out.append(
                {
                    "file": p.name,
                    "path": str(p),
                    "size_bytes": st.st_size,
                    "modified_ts": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                }
            )
        except OSError:
            continue
    return out


# ---------------------------------------------------------------------------
# Daily report response builder
# ---------------------------------------------------------------------------

def daily_report_response(
    *,
    report_dir: Path,
    get_conn_fn: Callable[[], Any],
    build_regime_context_fn: Callable[[Any], dict[str, Any]],
    pipeline_status_fn: Callable[..., dict[str, Any]],
    limit: int,
    include_markdown: bool,
    include_internal_paths: bool,
    include_admin_log_hints: bool,
) -> dict[str, Any]:
    """Build the daily report payload for admin/public APIs.

    Parameters
    ----------
    report_dir:
        Path to report JSON directory.
    get_conn_fn:
        Callable returning a sqlite3 Connection (to avoid circular imports).
    build_regime_context_fn:
        ``_report_build_regime_context(conn)`` callable.
    pipeline_status_fn:
        ``_pipeline_status_snapshot(log_lines=120)`` callable.
    limit:
        Max number of history entries.
    include_markdown:
        Whether to include the latest markdown text.
    include_internal_paths:
        Whether to expose file-system paths.
    include_admin_log_hints:
        Whether to suggest admin log endpoints in diagnostics.
    """
    latest_path, latest_payload = latest_daily_report_json(report_dir)
    if isinstance(latest_payload, dict):
        signals = latest_payload.get("signals")
        if isinstance(signals, dict):
            regime_ctx = signals.get("regime_context")
            needs_ma_matrix = not (
                isinstance(regime_ctx, dict)
                and isinstance(regime_ctx.get("ma_matrix"), list)
                and len(regime_ctx.get("ma_matrix") or []) > 0
            )
            comparison = regime_ctx.get("comparison") if isinstance(regime_ctx, dict) else None
            needs_comparison = not (
                isinstance(comparison, dict)
                and isinstance(comparison.get("series"), list)
                and len(comparison.get("series") or []) > 0
            )
            if needs_ma_matrix or needs_comparison:
                conn = get_conn_fn()
                try:
                    live_regime = build_regime_context_fn(conn)
                except Exception as exc:
                    LOG.warning("Failed to build live regime context: %s", exc)
                    live_regime = {}
                finally:
                    conn.close()
                if isinstance(live_regime, dict) and live_regime:
                    merged = dict(regime_ctx) if isinstance(regime_ctx, dict) else {}
                    for key in (
                        "asof_date",
                        "summary",
                        "llm_commentary",
                        "vix",
                        "ma_matrix",
                        "comparison",
                        "participation",
                        "overall",
                        "health",
                        "timeframes",
                        "levels",
                    ):
                        if key not in merged or not merged.get(key):
                            merged[key] = live_regime.get(key)
                    if not str(merged.get("source") or "").strip():
                        merged["source"] = "regime_context_live_patch"
                    signals["regime_context"] = merged
    pipeline = pipeline_status_fn(log_lines=120)
    detail: str | None = None
    detail_code: str | None = None
    detail_level: str | None = None
    detail_blocks_main_report = False
    log_hint = "/api/admin/logs?name=cron" if include_admin_log_hints else "server logs"
    if latest_payload is None:
        detail = "No report file found yet."
        detail_code = "report_missing"
        detail_level = "error"
        detail_blocks_main_report = True
    elif pipeline.get("active"):
        detail = (
            "daily_update is still running"
            f" (stage={pipeline.get('stage', 'unknown')}); "
            "generated_ts will advance after report stage completes."
        )
        detail_code = "pipeline_running"
        detail_level = "info"
    else:
        latest_run = pipeline.get("latest_run") or {}
        run_finished_ts = parse_iso_utc(latest_run.get("finished_ts")) if latest_run else None
        generated_ts = parse_iso_utc((latest_payload or {}).get("generated_ts"))
        if run_finished_ts is not None:
            if generated_ts is None:
                detail = (
                    "Latest ingest run finished, but latest report JSON has no generated_ts. "
                    f"Check {log_hint} for [REPORT] errors."
                )
                detail_code = "report_missing_generated_ts"
                detail_level = "warning"
            elif generated_ts < (run_finished_ts - dt.timedelta(seconds=60)):
                detail = (
                    "Latest ingest run finished at "
                    f"{run_finished_ts.replace(microsecond=0).isoformat()}, "
                    "but report generated_ts is still "
                    f"{generated_ts.replace(microsecond=0).isoformat()}. "
                    f"Report output is stale; check {log_hint} for [REPORT] errors."
                )
                detail_code = "report_stale"
                detail_level = "warning"
        email_block = latest_payload.get("email", {}) if isinstance(latest_payload, dict) else {}
        if detail is None and isinstance(email_block, dict):
            attempted = bool(email_block.get("attempted"))
            sent = bool(email_block.get("sent"))
            if attempted and not sent:
                error_msg = str(email_block.get("error") or "unknown SMTP error")
                detail = f"Report generated, but email delivery failed: {error_msg}"
                detail_code = "email_delivery_failed"
                detail_level = "warning"
    latest_md_path = report_dir / "daily_report_latest.md"
    md_text = ""
    if include_markdown and latest_md_path.exists():
        try:
            md_text = latest_md_path.read_text(encoding="utf-8")
        except Exception as exc:
            LOG.warning("Failed to read markdown file %s: %s", latest_md_path.name, exc)
            md_text = ""

    history = daily_report_history(report_dir, limit=limit)
    if not include_internal_paths:
        for row in history:
            row.pop("path", None)

    payload: dict[str, Any] = {
        "ok": latest_payload is not None,
        "latest": latest_payload or {},
        "history": history,
        "detail": detail,
        "detail_code": detail_code,
        "detail_level": detail_level,
        "detail_blocks_main_report": detail_blocks_main_report,
        "pipeline": {
            "active": pipeline.get("active"),
            "stage": pipeline.get("stage"),
            "latest_run": pipeline.get("latest_run"),
            "run_log_path": pipeline.get("run_log_path"),
        },
        "latest_markdown": md_text,
    }
    if include_internal_paths:
        payload["report_dir"] = str(report_dir)
        payload["latest_file"] = str(latest_path) if latest_path else None
    else:
        payload["pipeline"] = {
            "active": pipeline.get("active"),
            "stage": pipeline.get("stage"),
        }
    return payload
