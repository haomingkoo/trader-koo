"""Signals for docs/research/hypotheses-v1.md (pre-registered, implemented as written).

Every function reads only the ``AsOfView``: prices through the signal date and SEC
facts filed before it (the usable-from-the-next-session cutoff). Interpretations of
the places where the registration text is silent are listed in the PR and noted
next to the code that makes them.

Share counts. Raw cover counts sometimes carry a wrong scale; ``accept_share_count`` rejects a
count more than 3x away from the median of the nine filings before it and uses the next older one.

Market cap. ``price_daily.close`` is stored split-adjusted to today, so the quote
actually printed on the signal date t is ``close_stored(t) * prod(ratio of splits with
ex-date > t)``. The share count S of the latest filing (filed_date f < t) is on the
split basis of f, and becomes today's-quote-comparable by ``prod(ratio of splits with
f < ex-date <= t)``, the only corporate actions with ex-date <= t. The product of the
two is the market cap on the quoted basis of day t; un-adjusting the stored close
recovers a number that was public on t, it does not bring in a later price.
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np
import pandas as pd

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.pit import Fact
from trader_koo.research.settings import get as _settings

from .data import AsOfView

# Pre-registered signal definitions. These numbers are part of the registration
# (docs/research/hypotheses-v1.md), not tunables: changing one makes a different
# hypothesis and needs a new version of the document, so they are not in config/research.toml.
WINSOR = 3.0                     # H1 Score: "winsorised at plus or minus 3"
YEAR_AGO_LAG_DAYS = 320          # H2 "year-on-year": TTM ending at or before P - 320d is the one ending near P - 365d
YEAR_SPAN_DAYS = (350, 380)      # H2/H3: accepted distance between two TTM period ends one year apart
QUARTER_LAG_DAYS = 46            # H2 "one quarter earlier": TTM ending at or before P - 46d is the one near P - 91d
QUARTER_SPAN_DAYS = (76, 106)    # H2: accepted distance between TTM period ends one quarter apart
VOL_WINDOW = 252                 # Reference runs: "lowest 252-session volatility"
MOM_LOOKBACK = 252               # H3 Rule and Reference runs: "from 252 to 21 sessions before the signal date"
MOM_SKIP = 21
SHORT_FLOAT_TOP_DECILE = 0.9     # H5 Rule: "top decile of short float"
H4_RULES = (5, 5, 0)             # H4 Rule: 5 largest, annual schedule: (top_n, buffer_n, min_hold) = 5 names, no buffer, no min hold

_B = _settings().bench
SHARE_CHECK_FILINGS = _B.share_check_filings
SHARE_CHECK_MIN_PRIOR = _B.share_check_min_prior
SHARE_SCALE_LIMIT = _B.share_scale_limit
SHARE_HISTORY_ROWS = _B.share_history_rows
_END_OF_TIME = "9999-12-31"
_PRICE_BASIS = "COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only'"


def zscores(x: pd.DataFrame, clip: float | None = None) -> pd.DataFrame:
    """Cross-sectional z-scores per column (population std), optionally clipped at +-clip."""
    sd = x.std(ddof=0).replace(0.0, np.nan)
    z = (x - x.mean()) / sd
    return z if clip is None else z.clip(-clip, clip)


def _days(a: str, b: str) -> int:
    return (dt.date.fromisoformat(a) - dt.date.fromisoformat(b)).days


def _shift(day: str, days: int) -> str:
    return (dt.date.fromisoformat(day) - dt.timedelta(days=days)).isoformat()


def split_factor(splits: list[tuple[str, float]], after: str, through: str = _END_OF_TIME) -> float:
    """Product of share-count ratios of splits with ``after`` < ex-date <= ``through``."""
    return float(np.prod([r for d, r in splits if after < d <= through])) if splits else 1.0


def accept_share_count(history: list[tuple[str, float]], splits: list[tuple[str, float]],
                       rejected: list) -> tuple[float, str] | None:
    """Newest filed share count that passes the scale check; ``history`` is (filed_date, count), newest first.

    Raw XBRL cover counts sometimes carry a wrong scale (AEP, QCOM, GRMN filed 1000x). A count is
    rejected when, against the median of the up to nine filings before it (each moved onto its
    basis with splits whose ex-date falls between that filing and this one), it is more than
    3x larger or smaller. The next older count is then tried. A count with fewer than two earlier
    filings cannot be checked and is accepted (the caller counts these). Rejected counts are
    appended to ``rejected``.
    """
    for i, (filed, count) in enumerate(history):
        prior = [c * split_factor(splits, f, filed) for f, c in history[i + 1:i + 1 + SHARE_CHECK_FILINGS]]
        if len(prior) < SHARE_CHECK_MIN_PRIOR or SHARE_SCALE_LIMIT ** -1 <= count / np.median(prior) <= SHARE_SCALE_LIMIT:
            return count, filed
        rejected.append((filed, count))
    return None


def filed_share_counts(data: AsOfView, ticker: str) -> list[tuple[str, float]]:
    """One share count per filing date usable at ``as_of`` (the later period end wins), newest first."""
    tags = [t for _, t in cfg.CONCEPTS["shares_outstanding"]]
    rows = data.conn.execute(
        f"SELECT f.filed_date, f.value FROM sec_fundamentals f JOIN sec_cik_map m ON m.cik = f.cik "
        f"WHERE m.ticker = ? AND f.tag IN ({','.join('?' * len(tags))}) AND f.filed_date <= ? "
        f"ORDER BY f.filed_date DESC, f.period_end DESC, f.accession DESC LIMIT ?",
        (ticker.upper(), *tags, data.filing_cutoff, SHARE_HISTORY_ROWS)).fetchall()
    out: list[tuple[str, float]] = []
    for filed, value in rows:
        if not value > 0:  # zero, negative or NaN: not a share count; counted, never used
            data.diag.note("share_counts_nonpositive_ignored", (ticker.upper(), filed, float(value)))
            continue
        if not out or out[-1][0] != filed:
            out.append((filed, float(value)))
    return out[:SHARE_CHECK_FILINGS + 2]


def latest_filed_shares(data: AsOfView, ticker: str) -> tuple[float, str] | None:
    """(share count, filed_date) of the latest filed share count usable at ``as_of`` that passes the scale check.

    Ties on filed_date go to the later period end (the cover-page count over the balance-sheet count).
    Off-scale counts are noted on ``data.diag`` as ``share_counts_rejected_off_scale``; a count that
    could not be checked (too few earlier filings) is counted as ``share_counts_accepted_unchecked``.
    """
    history = filed_share_counts(data, ticker)
    rejected: list = []
    found = accept_share_count(history, data.splits.get(ticker, []), rejected)
    for filed, count in rejected:
        data.diag.note("share_counts_rejected_off_scale", (ticker.upper(), filed, count))
    if found is not None:
        later = [h for h in history if h[0] < found[1]]
        if len(later) < SHARE_CHECK_MIN_PRIOR:
            data.diag.count("share_counts_accepted_unchecked")
    return found


def stored_closes(data: AsOfView) -> pd.Series:
    """Stored (split-adjusted to today) close on the signal date, per ticker."""
    rows = data.conn.execute(
        f"SELECT ticker, close FROM price_daily WHERE date = ? AND {_PRICE_BASIS} AND close > 0",
        (data.as_of,)).fetchall()
    return pd.Series({t: c for t, c in rows}, dtype=float)


def market_caps(data: AsOfView, universe: list[str]) -> pd.Series:
    """Market cap on the quoted basis of the signal date; NaN without a close or a usable share count.

    Both NaN causes are counted on ``data.diag`` (name-reviews, not distinct names).
    """
    closes = stored_closes(data)
    out = {}
    for t in universe:
        shares = latest_filed_shares(data, t)
        if shares is None or t not in closes.index:
            data.diag.count("market_cap_nan_no_usable_share_count" if shares is None else "market_cap_nan_no_close_on_date")
            out[t] = np.nan
            continue
        count, filed = shares
        splits = data.splits.get(t, [])
        quote = closes[t] * split_factor(splits, data.as_of)
        out[t] = quote * count * split_factor(splits, filed, data.as_of)
    return pd.Series(out, index=universe, dtype=float)


def _val(f: Fact | None) -> float:
    return np.nan if f is None else f.value


def gross_profit_to_assets(data: AsOfView, t: str) -> float:
    gp, assets = data.ttm(t, "gross_profit"), data.fundamentals(t, "assets", "instant")
    return _val(gp) / assets.value if gp and assets and assets.value > 0 else np.nan


def liabilities_to_assets(data: AsOfView, t: str) -> float:
    liab, assets = data.fundamentals(t, "liabilities", "instant"), data.fundamentals(t, "assets", "instant")
    return liab.value / assets.value if liab and assets and assets.value > 0 else np.nan


def free_cash_flow(data: AsOfView, t: str) -> float:
    """Operating cash flow TTM minus capital expenditure TTM."""
    ocf, capex = data.ttm(t, "operating_cash_flow"), data.ttm(t, "capex")
    return ocf.value - capex.value if ocf and capex else np.nan


def complete_rows(x: pd.DataFrame, diag, prefix: str) -> pd.DataFrame:
    """Rows of ``x`` with every input finite. The others are dropped (the signal scores them NaN) and counted.

    Counts, as name-reviews on ``diag``: ``<prefix>_input_missing:<column>`` per column,
    ``<prefix>_infinite_input_set_to_missing`` and ``<prefix>_unscored_missing_input`` (rows dropped).
    """
    diag.count(f"{prefix}_infinite_input_set_to_missing", int(np.isinf(x.to_numpy(float)).sum()))
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.columns:
        diag.count(f"{prefix}_input_missing:{col}", int(x[col].isna().sum()))
    out = x.dropna()
    diag.count(f"{prefix}_unscored_missing_input", len(x) - len(out))
    return out


def quality_at_fair_price(params: dict):
    """H1: mean of z(gross profit / assets), z(-liabilities / assets), z(FCF / market cap), each clipped at +-3.

    The z-scores are taken over the names that have all three inputs; the others score NaN
    (HYPOTHESES "Excluded"); the count and the missing input are reported.
    """
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        mcap = market_caps(data, universe)
        x = complete_rows(pd.DataFrame({
            "gp_assets": [gross_profit_to_assets(data, t) for t in universe],
            "neg_leverage": [-liabilities_to_assets(data, t) for t in universe],
            "fcf_yield": [free_cash_flow(data, t) for t in universe] / mcap.where(mcap > 0).to_numpy(),
        }, index=universe), data.diag, "h1")
        return zscores(x, WINSOR).mean(axis=1).reindex(universe)

    return signal


def _year_ago(data: AsOfView, t: str, concept: str, now: Fact) -> Fact | None:
    prior = data.ttm(t, concept, _shift(now.period_end, YEAR_AGO_LAG_DAYS))
    return prior if prior and YEAR_SPAN_DAYS[0] <= _days(now.period_end, prior.period_end) <= YEAR_SPAN_DAYS[1] else None


def ttm_growth(data: AsOfView, t: str, concept: str, *, quarters_back: int = 0) -> float:
    """Year-on-year growth of TTM ``concept`` ending ``quarters_back`` (0 or 1) quarters before the newest.

    Growth is (now - year_ago) / |year_ago|, NaN when the year-ago TTM is zero or missing.
    A base of the other sign is rare for revenue; for EPS it keeps "improving" positive.
    """
    now = data.ttm(t, concept)
    if now is None:
        return np.nan
    if quarters_back:
        earlier = data.ttm(t, concept, _shift(now.period_end, QUARTER_LAG_DAYS))
        if earlier is None or not QUARTER_SPAN_DAYS[0] <= _days(now.period_end, earlier.period_end) <= QUARTER_SPAN_DAYS[1]:
            return np.nan
        now = earlier
    base = _year_ago(data, t, concept, now)
    if base is None or base.value == 0:
        return np.nan
    return (now.value - base.value) / abs(base.value)


def revenue_acceleration(data: AsOfView, t: str) -> float:
    """Revenue growth now minus the same measure one quarter earlier (needs 8 quarters of filings)."""
    return ttm_growth(data, t, "revenue") - ttm_growth(data, t, "revenue", quarters_back=1)


def accelerating_fundamentals(params: dict):
    """H2: mean of z(revenue growth acceleration) and z(year-on-year TTM diluted EPS growth)."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        x = complete_rows(pd.DataFrame({
            "rev_accel": [revenue_acceleration(data, t) for t in universe],
            "eps_growth": [ttm_growth(data, t, "eps_diluted") for t in universe],
        }, index=universe), data.diag, "h2")
        return zscores(x).mean(axis=1).reindex(universe)

    return signal


def momentum_confirmed_by_earnings(params: dict):
    """H3: 12-1 total return among names with positive year-on-year TTM diluted EPS growth."""
    lookback, skip = int(params.get("lookback", MOM_LOOKBACK)), int(params.get("skip", MOM_SKIP))

    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        px = data.close.reindex(columns=universe)
        if len(px) <= lookback:
            return pd.Series(np.nan, index=universe)
        mom = px.iloc[-1 - skip] / px.iloc[-1 - lookback] - 1.0
        growth = pd.Series([ttm_growth(data, t, "eps_diluted") for t in universe], index=universe)
        return mom.where(growth > 0)

    return signal


def market_cap_leaders(params: dict):
    """H4: score is market cap; the annual schedule keeps the top 5."""
    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        return market_caps(data, universe)

    return signal


def low_volatility(params: dict):
    """Reference: minus the standard deviation of the last 252 daily returns (complete windows only)."""
    window = int(params.get("window", VOL_WINDOW))

    def signal(as_of: str, universe: list[str], data: AsOfView) -> pd.Series:
        px = data.close.reindex(columns=universe)
        if len(px) <= window:
            return pd.Series(np.nan, index=universe)
        ret = px.iloc[-window - 1:].pct_change(fill_method=None).iloc[1:]
        return -ret.std(ddof=1).where(ret.notna().all())

    return signal


def short_float(conn: sqlite3.Connection, as_of: str, unparsed: list[tuple[str, str]]) -> pd.Series:
    """Finviz short float (fraction) from the latest snapshot taken before ``as_of``; prospective use only.

    Raises LookupError when there is no snapshot: an empty series would read as "nobody is shorted".
    A value that cannot be read (``-`` or empty) is left out and appended to ``unparsed`` as (ticker, text).
    """
    import json

    row = conn.execute("SELECT MAX(snapshot_ts) FROM finviz_fundamentals WHERE snapshot_ts < ?", (as_of,)).fetchone()
    if not row or not row[0]:
        raise LookupError(f"no finviz_fundamentals snapshot before {as_of}; short float is unavailable")
    out = {}
    for ticker, raw in conn.execute("SELECT ticker, raw_json FROM finviz_fundamentals WHERE snapshot_ts = ?", (row[0],)):
        text = json.loads(raw).get("Short Float", "")
        try:
            out[ticker] = float(text.rstrip("%")) / 100.0
        except ValueError:
            unparsed.append((ticker, text))
    return pd.Series(out, dtype=float)


def h5_fade_the_crowd(h1_targets: list[str], short_floats: pd.Series, universe: list[str],
                      kept_without_value: list[str]) -> list[str]:
    """H5: the H1 portfolio without names in the top decile of short float across the universe.

    The decile cut is the 90th percentile of short float among universe names that have a value;
    names at or above it are dropped and not replaced. Names with no value are kept (H5 rule) and
    appended to ``kept_without_value``. If no universe name has a value the cut cannot be made and
    this raises ValueError instead of returning the H1 portfolio unchanged.
    """
    known = short_floats.reindex(universe).dropna()
    if known.empty:
        raise ValueError("no universe name has a short float value; the H5 cut cannot be made")
    kept_without_value.extend(t for t in h1_targets if t not in known.index)
    cut = known.quantile(SHORT_FLOAT_TOP_DECILE)
    return [t for t in h1_targets if not (t in known.index and known[t] >= cut)]
