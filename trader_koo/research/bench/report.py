"""Metrics and the markdown table. Research only: nothing here is a performance claim."""
from __future__ import annotations

import numpy as np

from .portfolio import Result

TRADING_DAYS = 252
YEAR_CHARS = 4


def window_returns(equity: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Daily returns of equity[lo:hi+1]; equity[-1] before the path is the 1.0 base."""
    prev = equity[lo - 1] if lo > 0 else 1.0
    v = equity[lo:hi + 1]
    return v / np.concatenate([[prev], v[:-1]]) - 1.0


def stats(r: np.ndarray) -> dict:
    n = len(r)
    if n == 0:
        return {"cagr": None, "sharpe": None, "max_drawdown": None, "total_return": None}
    curve = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    sd = r.std(ddof=1) if n > 1 else 0.0
    return {
        "cagr": float(curve[-1] ** (TRADING_DAYS / n) - 1.0),
        "sharpe": float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else None,
        "max_drawdown": float((curve / peak - 1.0).min()),
        "total_return": float(curve[-1] - 1.0),
    }


def turnover(res: Result, equity: np.ndarray, lo_idx: int, hi_idx: int) -> float:
    """Annualised one-way turnover: traded notional / 2 / mean equity / years."""
    m = (res.trade_idx >= lo_idx) & (res.trade_idx <= hi_idx)
    years = (hi_idx - lo_idx + 1) / TRADING_DAYS
    return float(res.trade_notional[m].sum() / 2 / np.mean(equity) / years)


def hit_rate(cand: Result, ew: Result, lo_idx: int, hi_idx: int) -> float | None:
    """Share of held names whose holding-period return beat the equal-weight universe's."""
    ew_mean = {i: float(np.mean(r)) for i, r in zip(ew.period_idx, ew.period_returns) if len(r)}
    wins = total = 0
    for i, r in zip(cand.period_idx, cand.period_returns):
        if lo_idx <= i <= hi_idx and i in ew_mean and len(r):
            wins += int((r > ew_mean[i]).sum())
            total += len(r)
    return wins / total if total else None


def by_year(dates: list[str], r_by_series: dict[str, np.ndarray]) -> list[dict]:
    years = np.array([d[:YEAR_CHARS] for d in dates])
    rows = []
    for y in sorted(set(years)):
        m = years == y
        row = {"year": int(y), "sessions": int(m.sum())}
        for name, r in r_by_series.items():
            row[name] = float(np.prod(1.0 + r[m]) - 1.0)
        rows.append(row)
    return rows


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


def markdown(report: dict) -> str:
    lines = [f"# Bench: {report['candidate']} ({report['config_id']})", "",
             "Research output on historical data; not a claim of alpha or profitability.", ""]
    for name in ("dev", "sealed"):
        b = report.get(name)
        if not b:
            continue
        lines += [f"## {name} window {b['start']} to {b['end']}", "",
                  "| metric | value |", "|---|---|"]
        s = b["strategy"]
        rows = [
            ("CAGR (net, %s bps/side)" % report["cost_bps"], _pct(s["cagr"])),
            ("CAGR SPY total return", _pct(b["spy"]["cagr"])),
            ("CAGR equal-weight universe (net)", _pct(b["equal_weight"]["cagr"])),
            ("Excess vs SPY", _pct(b["excess_vs_spy"])),
            ("Excess vs equal-weight", _pct(b["excess_vs_equal_weight"])),
            ("Sharpe", "n/a" if s["sharpe"] is None else f"{s['sharpe']:.2f}"),
            ("Max drawdown", _pct(s["max_drawdown"])),
            ("Turnover (annual, one-way)", f"{b['turnover']:.2f}x"),
            ("Hit rate of holdings vs equal-weight", _pct(b["hit_rate"])),
            ("Years beating SPY", _pct(b["pct_years_beating_spy"])),
            ("Random-control percentile (n=%d)" % report["n_random"], f"{b['random']['percentile']:.1f}"),
            ("Random control CAGR p5 / median / p95",
             " / ".join(_pct(b["random"][k]) for k in ("p5", "p50", "p95"))),
            ("Stress CAGR (%s bps/side)" % report["stress_bps"], _pct(b["stress"]["cagr"])),
            ("Stress excess vs SPY", _pct(b["stress_excess_vs_spy"])),
        ]
        lines += [f"| {k} | {v} |" for k, v in rows]
        lines += ["", "| year | strategy | SPY | equal-weight | excess vs SPY |", "|---|---|---|---|---|"]
        for y in b["years"]:
            lines.append(f"| {y['year']} | {_pct(y['strategy'])} | {_pct(y['spy'])} | "
                         f"{_pct(y['equal_weight'])} | {_pct(y['strategy'] - y['spy'])} |")
        lines.append("")
    d = report["data"]
    lines += [f"Names lacking data: {d['entries_dropped_no_open']} entries skipped for no open price; "
              f"{d['flagged_price_rows']} price rows flagged in the window "
              f"({d['flagged_zeroed_or_nan']} treated as 0% or NaN, the rest keep their real move).", ""]
    lines += [f"- {c}" for c in report["caveats"]]
    return "\n".join(lines)
