"""What works and what doesn't — one diagnostic pass over a running Trader Koo.

`verify_deployment.py` is a fail-closed release gate: binary, needs an expected
SHA, raises on the first failing contract. This is the opposite. It reports the
state of every subsystem and keeps going, so a stale feed or an inert gate is
visible instead of being hidden behind whichever check failed first.

Usage::

    python -m trader_koo.scripts.system_check --base-url https://trader.kooexperience.com
    python -m trader_koo.scripts.system_check --db-path /data/trader_koo.db   # on the box
    python -m trader_koo.scripts.system_check --base-url ... --db-path ... --json

HTTP checks and database checks are independent: pass either, or both.
Exit code is 1 if anything reports FAIL, otherwise 0 (WARN does not fail).

ponytail: stdlib only (urllib + sqlite3) so it runs on the Railway container
without the app's virtualenv. Add a proper client if it ever needs auth flows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Feeds and how many hours of silence is tolerable before it is worth saying so.
# Weekday-only feeds get a weekend allowance rather than a special calendar.
_FRESHNESS: tuple[tuple[str, str, str, float], ...] = (
    ("price_daily", "SELECT MAX(date) FROM price_daily", "daily equity prices", 96.0),
    ("crypto_bars", "SELECT MAX(timestamp) FROM crypto_bars", "crypto feed", 1.0),
    ("finviz_fundamentals", "SELECT MAX(snapshot_ts) FROM finviz_fundamentals", "fundamentals", 96.0),
    ("polymarket_snapshots", "SELECT MAX(snapshot_ts) FROM polymarket_snapshots", "prediction markets", 1.0),
    ("hyperliquid_snapshots", "SELECT MAX(snapshot_ts) FROM hyperliquid_snapshots", "whale tracker", 1.0),
    ("hyperliquid_fills", "SELECT MAX(fill_date) FROM hyperliquid_fills", "whale fills", 96.0),
    ("options_iv", "SELECT MAX(snapshot_ts) FROM options_iv", "option chains", 96.0),
)


class Report:
    """Accumulates results so every check runs, unlike the release gate."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, area: str, name: str, state: str, detail: str) -> None:
        self.rows.append({"area": area, "check": name, "state": state, "detail": detail})

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["state"] == FAIL]

    def render(self) -> str:
        width = max((len(r["check"]) for r in self.rows), default=10)
        out: list[str] = []
        area = None
        for row in self.rows:
            if row["area"] != area:
                area = row["area"]
                out.append(f"\n{area}")
                out.append("-" * len(area))
            out.append(f"  {row['state']:4}  {row['check']:<{width}}  {row['detail']}")
        counts = {s: sum(1 for r in self.rows if r["state"] == s) for s in (PASS, WARN, FAIL)}
        out.append(f"\n{counts[PASS]} pass, {counts[WARN]} warn, {counts[FAIL]} fail")
        return "\n".join(out)


def _get(base_url: str, path: str, *, api_key: str | None = None, timeout: float = 45.0) -> tuple[int, Any, float]:
    headers = {"Accept": "application/json", "User-Agent": "trader-koo-system-check/1"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers=headers)
    started = dt.datetime.now(dt.timezone.utc)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
            try:
                return response.status, json.loads(body), elapsed
            except ValueError:
                return response.status, None, elapsed
    except urllib.error.HTTPError as exc:
        return exc.code, None, (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    except Exception as exc:  # network, DNS, timeout
        return 0, str(exc), (dt.datetime.now(dt.timezone.utc) - started).total_seconds()


def _age_hours(value: Any) -> float | None:
    """Hours since an ISO timestamp or YYYY-MM-DD date. None if unparseable."""
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    for parse in (dt.datetime.fromisoformat, lambda s: dt.datetime.strptime(s, "%Y-%m-%d")):
        try:
            stamp = parse(text)
        except (ValueError, TypeError):
            continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return (dt.datetime.now(dt.timezone.utc) - stamp).total_seconds() / 3600.0
    return None


def check_http(report: Report, base_url: str, api_key: str | None) -> None:
    code, payload, elapsed = _get(base_url, "/api/health")
    ok = code == 200 and isinstance(payload, dict) and payload.get("ok") is True
    report.add("HTTP", "health", PASS if ok else FAIL, f"HTTP {code} in {elapsed:.2f}s")

    for name, path, slow in (
        ("status", "/api/status", 3.0),
        ("report (ui view)", "/api/daily-report?view=ui", 3.0),
        ("report (full)", "/api/daily-report", 5.0),
        ("paper summary", "/api/paper-trades/summary", 3.0),
        ("crypto summary", "/api/crypto/summary", 3.0),
        ("chart quick", "/api/dashboard/SPY/quick?months=12", 5.0),
    ):
        code, _, elapsed = _get(base_url, path)
        if code != 200:
            report.add("HTTP", name, FAIL, f"HTTP {code}")
        elif elapsed > slow:
            report.add("HTTP", name, WARN, f"HTTP 200 but {elapsed:.2f}s (over {slow:.0f}s)")
        else:
            report.add("HTTP", name, PASS, f"HTTP 200 in {elapsed:.2f}s")

    # An unauthenticated admin route must be refused; that guard is easy to lose.
    code, _, _ = _get(base_url, "/api/admin/agent-observability")
    report.add(
        "HTTP", "admin requires key",
        PASS if code in {401, 403} else FAIL,
        f"unauthenticated request got HTTP {code}",
    )
    if api_key:
        code, payload, _ = _get(base_url, "/api/admin/agent-observability", api_key=api_key)
        ok = code == 200 and isinstance(payload, dict) and payload.get("ok") is True
        report.add("HTTP", "admin accepts key", PASS if ok else FAIL, f"HTTP {code}")


def check_freshness(report: Report, conn: sqlite3.Connection) -> None:
    for table, query, label, budget in _FRESHNESS:
        try:
            row = conn.execute(query).fetchone()
        except sqlite3.Error as exc:
            report.add("Data freshness", label, FAIL, f"{table}: {exc}")
            continue
        age = _age_hours(row[0] if row else None)
        if age is None:
            report.add("Data freshness", label, FAIL, f"{table} is empty or unparseable")
        elif age > budget:
            report.add("Data freshness", label, FAIL, f"{table} last wrote {age:.1f}h ago (budget {budget:.0f}h)")
        else:
            report.add("Data freshness", label, PASS, f"{table} {age:.1f}h old")


def check_gates(report: Report, conn: sqlite3.Connection) -> None:
    """The risk and promotion gates that can be silently inert."""
    # Sector concentration: the map must actually resolve the universe.
    try:
        snapshot = conn.execute("SELECT MAX(snapshot_ts) FROM finviz_fundamentals").fetchone()[0]
        rows = conn.execute(
            "SELECT raw_json FROM finviz_fundamentals WHERE snapshot_ts = ?", (snapshot,)
        ).fetchall()
        total = len(rows)
        with_sector = 0
        for (raw,) in rows:
            try:
                obj = json.loads(raw or "{}")
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict) and str(obj.get("Sector") or "").strip() not in ("", "-"):
                with_sector += 1
        pct = (with_sector / total * 100.0) if total else 0.0
        state = PASS if pct >= 90.0 else (WARN if pct >= 50.0 else FAIL)
        report.add(
            "Gates", "sector coverage", state,
            f"{with_sector}/{total} tickers ({pct:.0f}%) carry a Sector — below this the "
            f"concentration gate cannot evaluate",
        )
    except sqlite3.Error as exc:
        report.add("Gates", "sector coverage", FAIL, str(exc))

    # ML observation: the activation gate needs closed trades carrying a prediction.
    try:
        closed = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status != 'open'"
        ).fetchone()[0]
        scored = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE status != 'open' "
            "AND ml_predicted_win_prob IS NOT NULL"
        ).fetchone()[0]
        if closed and scored == 0:
            state, detail = FAIL, f"0 of {closed} closed trades carry a prediction — observation mode is not recording"
        elif scored < 20:
            state, detail = WARN, f"{scored}/20 closed trades scored (activation gate needs 20)"
        else:
            state, detail = PASS, f"{scored} closed trades scored"
        report.add("Gates", "ML observation", state, detail)
    except sqlite3.Error as exc:
        report.add("Gates", "ML observation", WARN, f"unavailable: {exc}")

    # Champion promotion: a registry that has never produced an eligible run.
    try:
        total = conn.execute("SELECT COUNT(*) FROM ml_validation_runs").fetchone()[0]
        eligible = conn.execute(
            "SELECT COUNT(*) FROM ml_validation_runs WHERE champion_eligible = 1"
        ).fetchone()[0]
        if total == 0:
            report.add("Gates", "ML promotion", WARN, "no validation runs recorded yet")
        elif eligible == 0:
            report.add("Gates", "ML promotion", WARN, f"0 of {total} runs ever reached champion_eligible")
        else:
            report.add("Gates", "ML promotion", PASS, f"{eligible}/{total} runs eligible")
    except sqlite3.Error as exc:
        report.add("Gates", "ML promotion", WARN, f"unavailable: {exc}")


def check_campaign(report: Report, conn: sqlite3.Connection) -> None:
    try:
        rows = conn.execute("SELECT campaign_id, status FROM paper_campaigns").fetchall()
    except sqlite3.Error as exc:
        report.add("Paper campaign", "campaigns", FAIL, str(exc))
        return
    active = [r[0] for r in rows if str(r[1]) == "active"]
    report.add(
        "Paper campaign", "active campaign",
        PASS if len(active) == 1 else WARN,
        f"{len(active)} active ({', '.join(active) or 'none'}) of {len(rows)} total",
    )
    for campaign_id in active:
        try:
            snap = conn.execute(
                "SELECT snapshot_date, equity, total_pnl_pct, open_trades, closed_trades_total "
                "FROM paper_portfolio_snapshots WHERE campaign_id = ? "
                "ORDER BY snapshot_date DESC LIMIT 1",
                (campaign_id,),
            ).fetchone()
        except sqlite3.Error as exc:
            report.add("Paper campaign", f"{campaign_id} snapshot", FAIL, str(exc))
            continue
        if not snap:
            report.add("Paper campaign", f"{campaign_id} snapshot", WARN, "no snapshot recorded")
            continue
        age = _age_hours(snap[0])
        state = PASS if (age is not None and age <= 96.0) else WARN
        report.add(
            "Paper campaign", f"{campaign_id} snapshot", state,
            f"{snap[0]} ({age:.0f}h ago) equity={snap[1]:,.0f} "
            f"total={snap[2]:+.2f}% open={snap[3]} closed={snap[4]}",
        )


def check_storage(report: Report, db_path: Path) -> None:
    try:
        size_mb = db_path.stat().st_size / 1_048_576
    except OSError as exc:
        report.add("Storage", "database", FAIL, str(exc))
        return
    report.add("Storage", "database", PASS, f"{size_mb:,.0f} MB")

    volume = db_path.parent
    for name in ("reports", "backups", "logs"):
        directory = volume / name
        if not directory.is_dir():
            continue
        files = [p for p in directory.iterdir() if p.is_file()]
        total_mb = sum(p.stat().st_size for p in files) / 1_048_576
        # ponytail: flat 500 MB line per directory. Tune if a real budget appears.
        state = WARN if total_mb > 500 else PASS
        report.add("Storage", name, state, f"{len(files)} files, {total_mb:,.0f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Report what works and what does not.")
    parser.add_argument("--base-url", help="e.g. https://trader.kooexperience.com")
    parser.add_argument("--db-path", help="e.g. /data/trader_koo.db")
    parser.add_argument("--api-key", help="X-API-Key, to also check the admin surface")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if not args.base_url and not args.db_path:
        parser.error("pass --base-url, --db-path, or both")

    report = Report()
    if args.base_url:
        check_http(report, args.base_url, args.api_key)
    if args.db_path:
        path = Path(args.db_path)
        if not path.exists():
            report.add("Database", "open", FAIL, f"{path} does not exist")
        else:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                check_freshness(report, conn)
                check_gates(report, conn)
                check_campaign(report, conn)
                check_storage(report, path)
            finally:
                conn.close()

    print(json.dumps(report.rows, indent=2) if args.json else report.render())
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
