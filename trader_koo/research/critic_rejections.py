"""What critic-rejected paper candidates would have done, check by check.

Pre-registered in ``docs/research/critic-rejections-v1.md``; the constants
below are that registration's frozen rules, not settings. The database is
opened read-only and nothing is written. Run it on a copy of production only.

    python -m trader_koo.research.critic_rejections --db COPY.db \\
        --campaign-id paper-v2 --out critic_rejections.md
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trader_koo.paper_trade.chronology import next_scheduled_session_after
from trader_koo.paper_trade.decision import direction_from_row

REGISTRATION = "docs/research/critic-rejections-v1.md"
HORIZON_BARS = 10
COST_PER_SIDE = 10 / 10_000  # 10 bps of price on entry and on exit
BASIS_TOLERANCE = 0.02
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261101
CI_PERCENTILES = (2.5, 97.5)
MIN_CLOSED_N = 30
FAIL_PATTERN = re.compile(r"^FAIL \[([^\]]+)\]:")
DIRECTIONS = {"long": 1, "short": -1}


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Candidate:
    report_run_id: str
    report_date: str
    ticker: str
    direction: str
    stop: float
    target: float
    reference: float
    report_close: float | None
    approved: bool
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class Outcome:
    status: str  # "closed", "open", or an exclusion reason
    gross_r: float | None = None
    net_r: float | None = None
    exit_reason: str | None = None


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def failed_checks(critic_reasons: list[str]) -> tuple[str, ...]:
    return tuple(m.group(1) for m in (FAIL_PATTERN.match(str(r)) for r in critic_reasons) if m)


def _direction(candidate_row: dict) -> tuple[str, str]:
    stored = str(candidate_row.get("call_direction") or "")
    if stored in DIRECTIONS:
        return stored, "call_direction"
    return direction_from_row(candidate_row), "direction_from_row"


def load_candidates(conn: sqlite3.Connection, campaign_id: str) -> tuple[list[Candidate], Counter, Counter]:
    """Rule set "Data" and rules 1-3: the in-scope decisions, plus counted exclusions."""
    rows = conn.execute(
        """SELECT d.report_run_id,d.report_date,d.ticker,d.stop_loss,d.target_price,
                  d.inputs_json,d.critic_outcome_json
           FROM paper_candidate_decisions d
           JOIN paper_decision_sets s
             ON s.report_run_id=d.report_run_id AND s.campaign_id=d.campaign_id
           WHERE d.campaign_id=? AND s.is_canonical=1 AND s.status='sealed'
           ORDER BY d.generated_ts,d.report_run_id,d.candidate_rank""",
        (campaign_id,),
    ).fetchall()
    excluded: Counter = Counter()
    sources: Counter = Counter()
    seen: set[tuple[str, str, str]] = set()
    out: list[Candidate] = []
    for run_id, report_date, ticker, stop, target, inputs_json, critic_json in rows:
        critic = json.loads(critic_json or "{}")
        if not critic:
            excluded["not_reviewed_by_critic"] += 1
            continue
        if not isinstance(critic.get("critic_reasons"), list) or "approved" not in critic:
            excluded["critic_outcome_without_check_detail"] += 1
            continue
        if stop is None or target is None:
            excluded["no_stop_or_target"] += 1
            continue
        inputs = json.loads(inputs_json or "{}")
        candidate_row = inputs.get("candidate") or {}
        direction, source = _direction(candidate_row)
        if direction not in DIRECTIONS:
            excluded["direction_not_long_or_short"] += 1
            continue
        reference = (inputs.get("context") or {}).get("entry_price")
        reference = candidate_row.get("close") if reference is None else reference
        if reference is None:
            excluded["no_reference_price"] += 1
            continue
        stop, target, reference = float(stop), float(target), float(reference)
        ordered = (stop < reference < target) if direction == "long" else (target < reference < stop)
        if not ordered:
            excluded["geometry_mismatch"] += 1
            continue
        key = (str(report_date), str(ticker), direction)
        if key in seen:
            excluded["duplicate_of_earlier_canonical_run"] += 1
            continue
        seen.add(key)
        sources[source] += 1
        out.append(Candidate(
            report_run_id=str(run_id), report_date=str(report_date), ticker=str(ticker),
            direction=direction, stop=stop, target=target, reference=reference,
            report_close=None if candidate_row.get("close") is None else float(candidate_row["close"]),
            approved=critic["approved"] is True, failed_checks=failed_checks(critic["critic_reasons"]),
        ))
    return out, excluded, sources


def simulate(direction: str, bars: list[Bar], stop: float, target: float, risk: float) -> Outcome:
    """Rules 7 and 8 on a complete window: enter at the first open, first touch wins, stop wins ties."""
    sign = DIRECTIONS[direction]
    entry = bars[0].open

    def beyond(price: float, level: float, toward_stop: bool) -> bool:
        return sign * (price - level) <= 0 if toward_stop else sign * (price - level) >= 0

    exit_price, reason = bars[-1].close, "horizon_close"
    for bar in bars:
        worst, best = (bar.low, bar.high) if sign == 1 else (bar.high, bar.low)
        if beyond(bar.open, stop, True):
            exit_price, reason = bar.open, "stop_gap_open"
        elif beyond(bar.open, target, False):
            exit_price, reason = bar.open, "target_gap_open"
        elif beyond(worst, stop, True):
            exit_price, reason = stop, "stop"
        elif beyond(best, target, False):
            exit_price, reason = target, "target"
        else:
            continue
        break
    gross = sign * (exit_price - entry) / risk
    net = gross - COST_PER_SIDE * (entry + exit_price) / risk
    return Outcome("closed", gross, net, reason)


def evaluate(conn: sqlite3.Connection, c: Candidate) -> Outcome:
    """Rules 4-6: point-in-time window from bars dated after the report date."""
    report_bar = conn.execute(
        "SELECT close FROM price_daily WHERE ticker=? AND date=?", (c.ticker, c.report_date)
    ).fetchone()
    if report_bar is None or report_bar[0] is None or not c.report_close:
        return Outcome("basis_unverifiable")
    if abs(float(report_bar[0]) / c.report_close - 1) > BASIS_TOLERANCE:
        return Outcome("basis_mismatch")
    rows = conn.execute(
        """SELECT date,open,high,low,close,basis_status FROM price_daily
           WHERE ticker=? AND date>? ORDER BY date LIMIT ?""",
        (c.ticker, c.report_date, HORIZON_BARS),
    ).fetchall()
    if not rows or rows[0][0] != next_scheduled_session_after(c.report_date):
        return Outcome("no_entry_bar")
    if len(rows) < HORIZON_BARS:
        return Outcome("open")
    if any(basis == "unresolved" for *_, basis in rows):
        return Outcome("unresolved_price_basis")
    if any(v is None for _, *ohlc, _ in rows for v in ohlc):
        return Outcome("incomplete_bar")
    bars = [Bar(d, float(o), float(h), float(lo), float(cl)) for d, o, h, lo, cl, _ in rows]
    return simulate(c.direction, bars, c.stop, c.target, abs(c.reference - c.stop))


def _cluster_draws(dates: list[str], values: list[float], rng: np.random.Generator) -> np.ndarray:
    """Bootstrap means resampling report dates (rule 9)."""
    keys = sorted(set(dates))
    index = {d: i for i, d in enumerate(keys)}
    sums, counts = np.zeros(len(keys)), np.zeros(len(keys))
    for d, v in zip(dates, values):
        sums[index[d]] += v
        counts[index[d]] += 1
    picks = rng.integers(0, len(keys), size=(BOOTSTRAP_RESAMPLES, len(keys)))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)


def _ci(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(draws, CI_PERCENTILES)
    return float(low), float(high)


def summarize(group: list[tuple[Candidate, Outcome]], approved: list[tuple[Candidate, Outcome]]) -> dict:
    closed = [(c, o) for c, o in group if o.status == "closed"]
    base = [(c, o) for c, o in approved if o.status == "closed"]
    out = {"n_closed": len(closed), "n_open": sum(o.status == "open" for _, o in group)}
    if not closed:
        return out
    dates = [c.report_date for c, _ in closed]
    net = [o.net_r for _, o in closed]
    wins = [float(o.net_r > 0) for _, o in closed]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mean_draws = _cluster_draws(dates, net, rng)
    out.update(
        mean_net_r=float(np.mean(net)), mean_net_r_ci=_ci(mean_draws),
        win_rate=float(np.mean(wins)), win_rate_ci=_ci(_cluster_draws(dates, wins, rng)),
        mean_gross_r=float(np.mean([o.gross_r for _, o in closed])),
        reading="descriptive" if min(len(closed), len(base)) >= MIN_CLOSED_N else "insufficient",
    )
    if base and group is not approved:
        base_draws = _cluster_draws([c.report_date for c, _ in base], [o.net_r for _, o in base], rng)
        out["diff_vs_approved"] = out["mean_net_r"] - float(np.mean([o.net_r for _, o in base]))
        out["diff_vs_approved_ci"] = _ci(mean_draws - base_draws)
    return out


def run(conn: sqlite3.Connection, campaign_id: str) -> dict:
    candidates, excluded, sources = load_candidates(conn, campaign_id)
    pairs = []
    for c in candidates:
        outcome = evaluate(conn, c)
        if outcome.status in ("closed", "open"):
            pairs.append((c, outcome))
        else:
            excluded[outcome.status] += 1
    approved = [p for p in pairs if p[0].approved]
    rejected = [p for p in pairs if not p[0].approved]
    groups = {"approved": summarize(approved, approved), "rejected (any check)": summarize(rejected, approved)}
    for name in sorted({n for c, _ in rejected for n in c.failed_checks}):
        members = [p for p in rejected if name in p[0].failed_checks]
        summary = summarize(members, approved)
        summary["n_sole_failure"] = sum(p[0].failed_checks == (name,) for p in members)
        groups[f"check: {name}"] = summary
    return {"campaign_id": campaign_id, "groups": groups, "excluded": dict(sorted(excluded.items())),
            "direction_sources": dict(sorted(sources.items())),
            "exit_reasons": dict(sorted(Counter(o.exit_reason for _, o in pairs if o.exit_reason).items()))}


def _fmt(value: float | None, ci: tuple[float, float] | None, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        text = f"{value:.0%}"
        return f"{text} [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else text
    text = f"{value:+.2f}R"
    return f"{text} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else text


def render(result: dict) -> str:
    lines = [
        f"# Critic rejection measurement v1: {result['campaign_id']}", "",
        f"Rules: `{REGISTRATION}`. Net of 10 bps per side, {HORIZON_BARS}-session windows from the next open, "
        f"95% intervals from {BOOTSTRAP_RESAMPLES:,} report-date bootstrap resamples (seed {BOOTSTRAP_SEED}).", "",
        f"n is small. A group is read only at {MIN_CLOSED_N} closed windows in it and in the approved group, "
        "and even then the figures are descriptive. Nothing here is a conclusion or a reason to change a threshold.", "",
        "| group | closed | open | sole failure | mean net R [95%] | win rate [95%] | mean gross R | net R minus approved [95%] | reading |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in result["groups"].items():
        lines.append("| " + " | ".join([
            name, str(g["n_closed"]), str(g["n_open"]), str(g.get("n_sole_failure", "-")),
            _fmt(g.get("mean_net_r"), g.get("mean_net_r_ci")), _fmt(g.get("win_rate"), g.get("win_rate_ci"), pct=True),
            _fmt(g.get("mean_gross_r"), None), _fmt(g.get("diff_vs_approved"), g.get("diff_vs_approved_ci")),
            g.get("reading", "insufficient"),
        ]) + " |")
    for title, key in (("Excluded", "excluded"), ("Direction source, before the price checks", "direction_sources"), ("Exit reasons", "exit_reasons")):
        lines += ["", f"## {title}", ""] + [f"- {k}: {v}" for k, v in result[key].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="path to a COPY of the production database")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out", help="Markdown file to write; stdout always gets the same text")
    args = parser.parse_args(argv)
    conn = connect_ro(args.db)
    try:
        text = render(run(conn, args.campaign_id))
    finally:
        conn.close()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
