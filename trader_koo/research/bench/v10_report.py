"""Markdown, ledger summary and JSON for a v10 run. Research only: nothing here is a performance claim."""
from __future__ import annotations

from .v9_report import jsonable as _jsonable

DIGITS = 4
DROP = frozenset({"exposure_path", "dates"})


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in DROP}
    return x


def jsonable(out: dict):
    return _jsonable(_strip({k: v for k, v in out.items() if k != "markdown"}))


def _row(name: str, m: dict, exposure: str = "") -> str:
    return (f"| {name} | {m['cagr']:+.2%} | {m['max_drawdown']:.1%} | {m['vol']:.1%} | {m['rv']:.2f} | "
            f"{m['sharpe']:.2f} | {m['calmar']:.2f} | {exposure} |")


def _block(key: str, r: dict) -> list[str]:
    lines = [f"### {key} ({r['book']} book)", "",
             "| series | CAGR | max DD | vol | return/vol | Sharpe over cash | CAGR/maxDD | mean exposure |",
             "|---|---|---|---|---|---|---|---|",
             _row("rule, 10 bps", r["strategy"], f"{r['mean_exposure']:.0%}"),
             _row("rule, 20 bps", r["stress"], f"{r['mean_exposure']:.0%}"),
             _row("buy-and-hold", r["buy_and_hold"], "100%"),
             _row(f"constant mix {r['mean_exposure']:.0%}", r["constant_mix"], f"{r['mean_exposure']:.0%}"), ""]
    q = r["shift_dd_quantiles"]
    lines += [f"Sessions invested {r['share_invested']:.0%}, {r['changes']} exposure changes. Timing controls: "
              f"max DD 5th/50th/95th pct {q[0]:.1%} / {q[1]:.1%} / {q[2]:.1%}; the rule's is shallower than "
              f"{r['shift']['pct_shallower_than']:.1f}% of shifts (p {r['shift']['p']:.4f}"
              + (f", Holm {r['adj_p']:.4f})." if "adj_p" in r else ")."),
              f"March 2020 (open 2020-02-20 to open 2020-03-24): rule {r['crash'][0]:+.1%}, "
              f"buy-and-hold {r['crash'][1]:+.1%}.", ""]
    if "gates" in r:
        lines += ["| gate | result |", "|---|---|"] + [f"| {g} | {'pass' if ok else 'FAIL'} |"
                                                       for g, ok in r["gates"].items()]
        lines += ["", f"**{key}: {r['verdict']}**", ""]
    lines += ["| year | rule | mix | rule return/vol | mix return/vol | rule CAGR/DD | mix CAGR/DD | b holds |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {y['year']} | {y['rule_return']:+.1%} | {y['mix_return']:+.1%} | {y['rule_rv']:.2f} | "
              f"{y['mix_rv']:.2f} | {y['rule_calmar']:.2f} | {y['mix_calmar']:.2f} | {'yes' if y['ok'] else 'no'} |"
              for y in r["years"]]
    lines += ["", f"Years where b holds: {r['year_share']:.0%}.", ""]
    return lines


def markdown(out: dict) -> str:
    c = out["config"]
    lines = [f"v10, {c['first_session']} to {c['last_interval']} (interval starts), seal {c['boundary']} not opened. "
             f"Indicators warm from {out['warm_session']}; {out['ftd_count_in_window']} uptrend-starting FTDs "
             "in the window.", ""]
    for k, r in out["results"].items():
        lines += _block(k, r)
    lines += ["## Variants (reported, outside the pass rule)", ""]
    for k, r in out["variants"].items():
        lines += _block(k, r)
    return "\n".join(lines)


def _brief(r: dict) -> dict:
    s = r["strategy"]
    return {"cagr": round(s["cagr"], DIGITS), "max_drawdown": round(s["max_drawdown"], DIGITS),
            "rv": round(s["rv"], DIGITS), "calmar": round(s["calmar"], DIGITS),
            "mean_exposure": round(r["mean_exposure"], DIGITS), "shift_p": r["shift"]["p"],
            "adj_p": r.get("adj_p"), "verdict": r.get("verdict", "variant")}


def summary(out: dict) -> dict:
    return {k: _brief(r) for k, r in {**out["results"], **out["variants"]}.items()}
