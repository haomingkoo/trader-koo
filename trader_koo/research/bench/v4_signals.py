"""Signals for docs/research/hypotheses-v4.md (pre-registered, implemented as written).

Same contract as the v1 signals: each reads only the ``AsOfView`` (prices through the
signal date, SEC facts filed before it). The adaptations forced by missing inputs are
stated in the registration; interpretations of places it is silent on are noted here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.pit import Fact

from . import hypotheses as hyp
from .data import AsOfView

# Pre-registered definitions (docs/research/hypotheses-v4.md). Changing one is a new version.
YEAR_SPAN_DAYS = hyp.YEAR_SPAN_DAYS      # "350 to 380 days before"
VALUE_QUANTILE = 0.8                     # H2 value set: top quintile of book-to-market
BM_TIEBREAK = 0.5                        # H2 score: F + 0.5 x book-to-market percentile
MA_SHORT = 50                            # H3 moving averages, sessions
MA_MID = 150
MA_LONG = 200
MA_RISING_LAG = 21                       # H3 rule 3: 200-session average above its value 21 sessions earlier
HIGH_LOW_WINDOW = 252                    # H3 rules 6 to 8
ABOVE_LOW = 1.30                         # H3 rule 6: at least 30% above the low
NEAR_HIGH = 0.75                         # H3 rule 7: within 25% of the high
RS_PERCENTILE = 0.70                     # H3 rule 8
EPS_GROWTH_MIN = 0.25                    # H3 earnings rule


def _instant(data: AsOfView, t: str, concept: str) -> Fact | None:
    return data.fundamentals(t, concept, "instant")


def year_ago_instant(data: AsOfView, t: str, concept: str, now: Fact) -> Fact | None:
    """Instant fact whose period end is 350 to 380 days before ``now``'s, from filings usable at ``as_of``.

    Concept tags are compared as in ``fundamentals_as_of``: the newest period end wins, earlier
    priority breaks ties; within one tag the latest filing of that period wins.
    """
    end = pd.Timestamp(now.period_end)
    lo = (end - pd.Timedelta(days=YEAR_SPAN_DAYS[1])).date().isoformat()
    hi = (end - pd.Timedelta(days=YEAR_SPAN_DAYS[0])).date().isoformat()
    best: Fact | None = None
    for _, tag in cfg.CONCEPTS[concept]:
        row = data.conn.execute(
            """SELECT f.tag, f.value, f.period_start, f.period_end, f.filed_date, f.form, f.accession
               FROM sec_fundamentals f JOIN sec_cik_map m ON m.cik = f.cik
               WHERE m.ticker = ? AND f.tag = ? AND f.period_start = '' AND f.filed_date <= ?
                 AND f.period_end BETWEEN ? AND ?
               ORDER BY f.period_end DESC, f.filed_date DESC, f.accession DESC LIMIT 1""",
            (t.upper(), tag, data.filing_cutoff, lo, hi)).fetchone()
        fact = Fact(*row) if row else None
        if fact and (best is None or fact.period_end > best.period_end):
            best = fact
    return best


# ---- H1 Magic Formula, adapted ------------------------------------------------------------------

def magic_formula_inputs(data: AsOfView, t: str, mcap: float) -> tuple[float, float]:
    """(earnings yield, adapted return on capital); NaN when any input is missing or EV <= 0."""
    ebit = data.ttm(t, "operating_income")
    debt, cash, assets = (_instant(data, t, c) for c in ("long_term_debt", "cash", "assets"))
    if ebit is None or debt is None or cash is None or assets is None or not np.isfinite(mcap):
        return np.nan, np.nan
    ev = mcap + debt.value - cash.value
    capital = assets.value - cash.value
    if ev <= 0 or capital <= 0:
        return np.nan, np.nan
    return ebit.value / ev, ebit.value / capital


def magic_formula(params: dict):
    """H1: minus the sum of the earnings-yield rank and the return-on-capital rank (rank 1 = highest)."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        mcap = hyp.market_caps(data, universe)
        x = hyp.complete_rows(pd.DataFrame(
            [magic_formula_inputs(data, t, mcap[t]) for t in universe],
            index=universe, columns=["earnings_yield", "return_on_capital"]), data.diag, "v4h1")
        if x.empty:
            return pd.Series(np.nan, index=universe)
        combined = x["earnings_yield"].rank(ascending=False) + x["return_on_capital"].rank(ascending=False)
        # Ties keep rank order by earnings yield: a tie-break smaller than one rank step.
        tiebreak = x["earnings_yield"].rank(ascending=True) / (len(x) + 1)
        return (-combined + tiebreak).reindex(universe)

    return signal


# ---- H2 Piotroski F-score, 7-signal adaptation --------------------------------------------------

def share_count_change(data: AsOfView, t: str) -> float:
    """Latest accepted filed share count over the one filed 350 to 380 days earlier, split adjusted, minus 1."""
    shares = hyp.latest_filed_shares(data, t)
    if shares is None:
        return np.nan
    count, filed = shares
    history = [h for h in hyp.filed_share_counts(data, t)
               if YEAR_SPAN_DAYS[0] <= (pd.Timestamp(filed) - pd.Timestamp(h[0])).days <= YEAR_SPAN_DAYS[1]]
    if not history:
        return np.nan
    old_filed, old_count = history[0]
    return count / (old_count * hyp.split_factor(data.splits.get(t, []), old_filed, filed)) - 1.0


def f_score(data: AsOfView, t: str) -> float:
    """Seven-signal F-score (0 to 7), NaN when an input is missing (long-term debt has its own rule)."""
    ni, cfo, rev = (data.ttm(t, c) for c in ("net_income", "operating_cash_flow", "revenue"))
    assets = _instant(data, t, "assets")
    if ni is None or cfo is None or rev is None or assets is None or assets.value <= 0:
        return np.nan
    ni_0, rev_0 = hyp._year_ago(data, t, "net_income", ni), hyp._year_ago(data, t, "revenue", rev)
    assets_0 = year_ago_instant(data, t, "assets", assets)
    if ni_0 is None or rev_0 is None or assets_0 is None or assets_0.value <= 0:
        return np.nan
    roa, roa_0 = ni.value / assets.value, ni_0.value / assets_0.value
    turn, turn_0 = rev.value / assets.value, rev_0.value / assets_0.value

    debt = _instant(data, t, "long_term_debt")
    debt_0 = year_ago_instant(data, t, "long_term_debt", debt) if debt else None
    if debt is None and data.conn.execute(  # files no long-term debt tag in either period: no increase
            "SELECT 1 FROM sec_fundamentals f JOIN sec_cik_map m ON m.cik = f.cik WHERE m.ticker = ? "
            f"AND f.tag IN ({','.join('?' * len(cfg.CONCEPTS['long_term_debt']))}) AND f.filed_date <= ? LIMIT 1",
            (t.upper(), *[g for _, g in cfg.CONCEPTS["long_term_debt"]], data.filing_cutoff)).fetchone() is None:
        leverage_down = True
    elif debt is not None and debt_0 is not None:
        leverage_down = debt.value / assets.value < debt_0.value / assets_0.value
    else:
        data.diag.count("v4h2_unscored_debt_in_one_period_only")
        return np.nan

    share_change = share_count_change(data, t)
    if np.isnan(share_change):
        return np.nan

    return float(sum((
        roa > 0, cfo.value > 0, roa > roa_0, cfo.value > ni.value,
        leverage_down, share_change <= 0, turn > turn_0,
    )))


def piotroski_value(params: dict):
    """H2: F-score plus 0.5 x book-to-market percentile, inside the top book-to-market quintile."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        mcap = hyp.market_caps(data, universe)
        equity = pd.Series([hyp._val(_instant(data, t, "equity")) for t in universe], index=universe)
        bm = (equity / mcap).where((equity > 0) & (mcap > 0)).dropna()
        data.diag.count("v4h2_no_book_to_market", len(universe) - len(bm))
        if bm.empty:
            return pd.Series(np.nan, index=universe)
        value = bm[bm >= bm.quantile(VALUE_QUANTILE)]
        f = pd.Series({t: f_score(data, t) for t in value.index}, dtype=float)
        data.diag.count("v4h2_unscored_missing_input", int(f.isna().sum()))
        pct = value.rank(pct=True)
        return (f + BM_TIEBREAK * pct).reindex(universe)

    return signal


# ---- H3 Trend Template with earnings growth -----------------------------------------------------

def trend_template_pass(px: pd.DataFrame) -> pd.Series:
    """Rules 1 to 7 on total-return-adjusted closes (the last row is the signal date)."""
    close = px.iloc[-1]
    ma50, ma150 = px.iloc[-MA_SHORT:].mean(), px.iloc[-MA_MID:].mean()
    ma200 = px.iloc[-MA_LONG:].mean()
    ma200_before = px.iloc[-MA_LONG - MA_RISING_LAG:-MA_RISING_LAG].mean()
    window = px.iloc[-HIGH_LOW_WINDOW:]
    low, high = window.min(), window.max()
    complete = px.iloc[-HIGH_LOW_WINDOW - 1:].notna().all() & px.iloc[-MA_LONG - MA_RISING_LAG:].notna().all()
    return (complete & (close > ma150) & (close > ma200) & (ma150 > ma200) & (ma200 > ma200_before)
            & (ma50 > ma150) & (ma50 > ma200) & (close > ma50)
            & (close >= ABOVE_LOW * low) & (close >= NEAR_HIGH * high))


def trend_template(params: dict):
    """H3: 252-session return among names passing the eight price rules and EPS growth >= 25%."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        px = data.close.reindex(columns=universe)
        if len(px) <= max(HIGH_LOW_WINDOW, MA_LONG + MA_RISING_LAG):
            return pd.Series(np.nan, index=universe)
        ret = px.iloc[-1] / px.iloc[-1 - HIGH_LOW_WINDOW] - 1.0
        rs_ok = ret.rank(pct=True) >= RS_PERCENTILE
        price_ok = trend_template_pass(px) & rs_ok
        passed = price_ok[price_ok].index
        growth = pd.Series({t: hyp.ttm_growth(data, t, "eps_diluted") for t in passed}, dtype=float)
        data.diag.count("v4h3_price_pass_no_eps_growth", int(growth.isna().sum()))
        keep = growth[growth >= EPS_GROWTH_MIN].index
        return ret.where(ret.index.isin(keep)).reindex(universe)

    return signal
