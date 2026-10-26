"""End-of-build data checks for the research price database.

Every number comes from a query on the database being checked. A check either
FAILs (the build stops) or REPORTs (a known limit that is listed, not hidden).
Thresholds are the constants below so a reviewer can argue with them in one place.

    python -m ops.research_db_checks --db-path ~/dev/.research/trader_koo_research.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CONTEXT_SYMBOLS = ("SPY", "QQQ", "^VIX", "^IRX")
BENCHMARK_SYMBOLS = ("SPY", "QQQ")
HISTORY_START_LATEST = "2010-01-05"
MIN_YEAR_PRICE_COVERAGE_PCT = 60.0  # member-days with a price row, every year
MIN_SEC_UNIVERSE_PCT = 40.0  # universe tickers whose CIK has at least one stored fact
SHARE_TAGS = ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")
SHARES_MAX = 5e11  # above this no listed company's share count is plausible
SHARES_MEDIAN_BAND = 100.0  # a value this many times above/below the CIK median is off-scale
LIST_LIMIT = 15
SPLIT_TYPES = ("split", "reverse_split")

MEMBER_DAYS_SQL = """
WITH sessions AS (SELECT DISTINCT date FROM price_daily WHERE ticker = 'SPY'),
members AS (
  SELECT s.date AS d, m.ticker AS t FROM sessions s JOIN sp500_membership m
    ON m.start_date <= s.date AND (m.end_date IS NULL OR m.end_date > s.date)
  {member_filter})
SELECT substr(d, 1, 4), COUNT(*), SUM(p.ticker IS NOT NULL)
FROM members LEFT JOIN price_daily p ON p.ticker = members.t AND p.date = members.d
GROUP BY 1 ORDER BY 1
"""


@dataclass
class Result:
    name: str
    ok: bool  # False only for a FAIL-class violation
    lines: list[str]
    kind: str = "FAIL"  # "FAIL" checks gate the build; "REPORT" checks only list


@dataclass
class Checks:
    results: list[Result] = field(default_factory=list)

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if not r.ok]


def _q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _universe(conn: sqlite3.Connection, tickers: list[str] | None) -> list[str]:
    rows = {r[0] for r in _q(conn, "SELECT DISTINCT ticker FROM sp500_membership")}
    return sorted(rows & set(tickers) if tickers else rows)


def check_price_coverage(conn: sqlite3.Connection, tickers: list[str] | None) -> Result:
    flt = ""
    params: tuple = ()
    if tickers:
        flt = f"WHERE m.ticker IN ({','.join('?' * len(tickers))})"
        params = tuple(tickers)
    rows = _q(conn, MEMBER_DAYS_SQL.format(member_filter=flt), params)
    lines, ok = [], bool(rows)
    if not rows:
        lines.append("no member-days: SPY sessions or sp500_membership are empty")
    for year, days, with_price in rows:
        pct = 100.0 * (with_price or 0) / days
        bad = pct < MIN_YEAR_PRICE_COVERAGE_PCT
        ok = ok and not bad
        lines.append(f"{year}  member_days={days:>7}  with_price={with_price or 0:>7}  {pct:5.1f}%"
                     + (f"  < floor {MIN_YEAR_PRICE_COVERAGE_PCT}%" if bad else ""))
    return Result(f"member-day price coverage by year (floor {MIN_YEAR_PRICE_COVERAGE_PCT}%)", ok, lines)


def check_benchmark_history(conn: sqlite3.Connection) -> Result:
    lines, ok = [], True
    for sym in CONTEXT_SYMBOLS:
        row = _q(conn, "SELECT MIN(date), MAX(date), COUNT(*) FROM price_daily WHERE ticker = ?", (sym,))[0]
        first, last, n = row
        bad = n == 0 or (sym in BENCHMARK_SYMBOLS and first > HISTORY_START_LATEST)
        ok = ok and not bad
        lines.append(f"{sym:<5} first={first} last={last} rows={n}" + ("  VIOLATION" if bad else ""))
    return Result(f"SPY/QQQ history start <= {HISTORY_START_LATEST}; context symbols present", ok, lines)


def _sec_ciks(conn: sqlite3.Connection, universe: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    marks = ",".join("?" * len(universe))
    for t, cik in _q(conn, f"SELECT ticker, cik FROM sec_cik_map WHERE ticker IN ({marks})", tuple(universe)):
        out[t].add(cik)
    return out


def check_sec_coverage(conn: sqlite3.Connection, universe: list[str]) -> Result:
    with_facts = {r[0] for r in _q(conn, "SELECT DISTINCT cik FROM sec_fundamentals")}
    mapped = _sec_ciks(conn, universe)
    have = [t for t in universe if mapped.get(t, set()) & with_facts]
    unmapped = [t for t in universe if t not in mapped]
    pct = 100.0 * len(have) / len(universe) if universe else 0.0
    ciks_loaded = len({c for t in have for c in mapped[t] & with_facts})
    status = dict(_q(conn, "SELECT status, COUNT(*) FROM sec_fetch_status GROUP BY status"))
    ok = bool(universe) and pct >= MIN_SEC_UNIVERSE_PCT
    lines = [
        f"universe tickers={len(universe)}  with facts={len(have)} ({pct:.1f}%)  floor {MIN_SEC_UNIVERSE_PCT}%",
        f"distinct CIKs with facts={ciks_loaded}  fetch status={status}",
        f"universe tickers with no SEC ticker mapping={len(unmapped)} (SEC maps current tickers only)",
    ]
    if status.get("error"):
        ok = False
        lines.append(f"sec_fetch_status holds {status['error']} 'error' CIKs: rerun to retry them")
    return Result("SEC fundamentals coverage of the universe", ok, lines)


def _share_rows(conn: sqlite3.Connection, universe: list[str]):
    cik_ticker = {c: t for t, cs in _sec_ciks(conn, universe).items() for c in cs}
    if not cik_ticker:
        return []
    marks = ",".join("?" * len(SHARE_TAGS))
    rows = _q(conn, f"SELECT cik, tag, period_end, value, accession FROM sec_fundamentals "
                    f"WHERE tag IN ({marks})", SHARE_TAGS)
    return [(cik_ticker[r[0]], *r) for r in rows if r[0] in cik_ticker]


def check_share_scale(conn: sqlite3.Connection, universe: list[str]) -> Result:
    rows = _share_rows(conn, universe)
    by_cik: dict[int, list[float]] = defaultdict(list)
    for _, cik, _, _, value, _ in rows:
        if 0 < value <= SHARES_MAX:
            by_cik[cik].append(value)
    medians = {c: statistics.median(v) for c, v in by_cik.items()}
    bad = []
    for ticker, cik, tag, period_end, value, accn in rows:
        med = medians.get(cik)
        reason = None
        if value <= 0:
            reason = "non_positive"
        elif value > SHARES_MAX:
            reason = "above_max"
        elif med and not (med / SHARES_MEDIAN_BAND <= value <= med * SHARES_MEDIAN_BAND):
            reason = "far_from_cik_median"
        if reason:
            bad.append((reason, ticker, cik, tag, period_end, value, accn))
    by_reason = defaultdict(int)
    for b in bad:
        by_reason[b[0]] += 1
    lines = [f"share-count facts checked={len(rows)}  off-scale={len(bad)}  by reason={dict(by_reason)}",
             f"off-scale = value <= 0, > {SHARES_MAX:g}, or outside {SHARES_MEDIAN_BAND:g}x of the CIK median",
             f"tickers affected={len({b[1] for b in bad})}"]
    for b in sorted(bad, key=lambda b: (-abs(b[5]), b[1]))[:LIST_LIMIT]:
        lines.append("  " + "  ".join(str(x) for x in b))
    if len(bad) > LIST_LIMIT:
        lines.append(f"  ... {len(bad) - LIST_LIMIT} more")
    return Result("SEC share counts off-scale (consumers must not use these as-is)", True, lines, "REPORT")


def check_multi_class(conn: sqlite3.Connection, universe: list[str]) -> Result:
    mapped = _sec_ciks(conn, universe)
    by_cik: dict[int, list[str]] = defaultdict(list)
    for t, ciks in mapped.items():
        for c in ciks:
            by_cik[c].append(t)
    shared = {c: sorted(ts) for c, ts in by_cik.items() if len(ts) > 1}
    lines = [f"CIKs shared by more than one universe ticker={len(shared)}",
             "SEC facts carry no share-class dimension: one CIK count covers all classes or only one."]
    for cik, ts in sorted(shared.items()):
        latest = _q(conn, "SELECT tag, period_end, value FROM sec_fundamentals WHERE cik = ? AND tag IN "
                    "('CommonStockSharesOutstanding','EntityCommonStockSharesOutstanding') "
                    "ORDER BY period_end DESC, tag LIMIT 1", (cik,))
        lines.append(f"  cik {cik}: {','.join(ts)}  latest share fact={latest[0] if latest else 'none'}")
    brk = [t for t in universe if t.startswith("BRK")]
    lines.append(f"BRK tickers in universe={brk}" + ("" if brk else "  (BRK-B not in this universe)"))
    return Result("multi-class share issues (BRK-B and others)", True, lines, "REPORT")


def check_missing_splits(conn: sqlite3.Connection, tickers: list[str] | None) -> Result:
    have = {(t, d, a) for t, d, a in _q(
        conn, "SELECT ticker, action_date, action_type FROM price_corporate_actions WHERE action_type IN (?,?)",
        SPLIT_TYPES)}
    missing: set[tuple] = set()
    seen = 0
    for ticker, start, end, evidence in _q(
            conn, "SELECT ticker, managed_start, managed_end, evidence_json FROM price_series_revisions"):
        if tickers and ticker not in tickers:
            continue
        doc = json.loads(evidence)
        for act in doc.get("vendor_action_ledger", []) + doc.get("normalization_actions", []):
            if act["action_type"] in SPLIT_TYPES and start <= act["action_date"] <= end:
                seen += 1
                key = (ticker, act["action_date"], act["action_type"])
                if key not in have:
                    missing.add(key + (act["value"],))
    lines = [f"vendor splits inside managed ranges checked={seen}  missing from price_corporate_actions={len(missing)}"]
    lines += [f"  {m}" for m in sorted(missing)]
    return Result("vendor splits in price_series_revisions but not in price_corporate_actions", True, lines, "REPORT")


def check_duplicates(conn: sqlite3.Connection) -> Result:
    probes = {
        "price_daily (upper(ticker), date)":
            "SELECT COUNT(*) FROM (SELECT 1 FROM price_daily GROUP BY upper(trim(ticker)), date HAVING COUNT(*) > 1)",
        "price_corporate_actions (upper(ticker), date, type, provider)":
            "SELECT COUNT(*) FROM (SELECT 1 FROM price_corporate_actions GROUP BY upper(trim(ticker)), "
            "action_date, action_type, provider HAVING COUNT(*) > 1)",
        "sp500_membership overlapping spells of one ticker":
            "SELECT COUNT(*) FROM sp500_membership a JOIN sp500_membership b ON a.ticker = b.ticker "
            "AND a.start_date < b.start_date AND (a.end_date IS NULL OR a.end_date > b.start_date)",
        "earnings_events (upper(ticker), report_date, source)":
            "SELECT COUNT(*) FROM (SELECT 1 FROM earnings_events GROUP BY upper(trim(ticker)), report_date, "
            "source HAVING COUNT(*) > 1)",
        "sec_fundamentals (cik, tag, unit, period_start, period_end, accession)":
            "SELECT COUNT(*) FROM (SELECT 1 FROM sec_fundamentals GROUP BY cik, tag, unit, period_start, "
            "period_end, accession HAVING COUNT(*) > 1)",
    }
    counts = {name: _q(conn, sql)[0][0] for name, sql in probes.items()}
    lines = [f"{n}: {c}" for n, c in counts.items()]
    return Result("duplicate keys = 0", not any(counts.values()), lines)


def check_flagged_rows(conn: sqlite3.Connection, tickers: list[str] | None) -> Result:
    flt, params = "", ()
    if tickers:
        flt, params = f" AND ticker IN ({','.join('?' * len(tickers))})", tuple(tickers)
    total = _q(conn, f"SELECT COUNT(*) FROM price_daily WHERE 1=1{flt}", params)[0][0]
    by_reason: dict[str, int] = defaultdict(int)
    for status, reason, n in _q(conn, "SELECT basis_status, unresolved_reason, COUNT(*) FROM price_daily "
                                f"WHERE basis_status != 'verified'{flt} GROUP BY 1, 2", params):
        try:
            codes = sorted({x["reason"] for x in json.loads(reason)}) if reason else []
        except (ValueError, TypeError, KeyError):
            codes = [f"unparsed:{reason}"]
        by_reason[f"{status}: {'+'.join(codes) or 'no reason recorded'}"] += n
    tr = _q(conn, f"SELECT COUNT(*), COUNT(DISTINCT ticker) FROM price_daily WHERE adjustment_basis = 'total_return'{flt}", params)[0]
    tr_flagged = _q(conn, "SELECT basis_status, COUNT(*) FROM price_daily WHERE adjustment_basis = "
                    f"'total_return'{flt} GROUP BY 1", params)
    lines = [f"price_daily rows={total}  not verified={sum(by_reason.values())}"]
    lines += [f"  {n:>8}  {k}" for k, n in sorted(by_reason.items(), key=lambda kv: -kv[1])]
    lines.append(f"rows on adjustment_basis=total_return={tr[0]} across {tr[1]} tickers; by basis_status={dict(tr_flagged)}")
    return Result("flagged price rows by reason (incl. total_return basis)", True, lines, "REPORT")


def run_checks(db_path: Path, tickers: list[str] | None = None) -> Checks:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        universe = _universe(conn, tickers)
        checks = Checks()
        checks.results += [
            check_price_coverage(conn, tickers),
            check_benchmark_history(conn),
            check_sec_coverage(conn, universe),
            check_duplicates(conn),
            check_share_scale(conn, universe),
            check_missing_splits(conn, tickers),
            check_flagged_rows(conn, tickers),
            check_multi_class(conn, universe),
        ]
        return checks
    finally:
        conn.close()


def render(checks: Checks) -> str:
    out = ["", "RESEARCH DB CHECKLIST"]
    for r in checks.results:
        tag = "REPORT" if r.kind == "REPORT" else ("PASS" if r.ok else "FAIL")
        out.append(f"[{tag}] {r.name}")
        out += [f"       {line}" for line in r.lines]
    failed = checks.failed
    out.append(f"RESULT: {'FAILED ' + str(len(failed)) + ' check(s)' if failed else 'all gating checks passed'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", required=True)
    p.add_argument("--tickers", help="restrict to a comma separated subset (test builds)")
    args = p.parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    checks = run_checks(Path(args.db_path).expanduser(), tickers)
    print(render(checks))
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
