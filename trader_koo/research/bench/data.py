"""Market data for the research bench: back-adjusted prices, universe, as-of access.

Prices are built from the dividend/split total-return chain in
``trader_koo.db.total_return``: ``adj_close`` compounds the total-return series
forward from each ticker's first row, so the value on day t depends only on rows
up to t. A dividend or split after t never changes anything a signal sees at t
(no back-adjustment). ``adj_open = adj_close * open / close`` puts opens and
closes on one basis with dividends inside the price.

Flagged rows. A flag only means the chain could not vouch for the move, not that
the move is wrong. Treating it as 0% hides real crashes (FISV -44%, CNC -40%) as
easily as real jumps. The policy here: a row flagged ``move_on_split_date`` (the
close is probably not split-adjusted) is a 0% day; ``bad_close`` is NaN; every
other flag (special dividend, unexplained move, off-calendar dividend) keeps the
real close-to-close move plus the dividend. Excluding the ticker for the window
would be hindsight (it uses the future jump to decide), so it is not done. Every
flagged row inside the run window is listed in the report so a human can judge it.
Nothing here writes to the database.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trader_koo.db.membership import members_on
from trader_koo.db.total_return import load_total_return_inputs, total_return_series
from trader_koo.fundamentals.pit import Fact, fundamentals_as_of, ttm_as_of
from trader_koo.research.bench import quality
from trader_koo.research.bench.diagnostics import Diagnostics
from trader_koo.research.bench.spinoffs import NEUTRALISED, load_spinoffs
from trader_koo.research.settings import get as _settings

_B = _settings().bench
BENCHMARK_TICKER = _B.benchmark_ticker
SPLIT_TYPES = ("split", "reverse_split")
SPLIT_ROUND = _B.split_ratio_round_digits
YEAR_CHARS = 4  # ISO date prefix width
KEEP_REAL_MOVE_FLAGS = _B.keep_real_move_flags


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def filing_cutoff(as_of: str) -> str:
    """Latest filed_date usable at the close of session ``as_of``: the calendar day before it.

    A filing dated d is usable from the first session after d. For a session
    ``as_of`` that is exactly filed_date < as_of, so a Friday filing is usable
    Monday, a weekend filing Monday, and a filing dated ``as_of`` itself not yet.
    """
    return (dt.date.fromisoformat(as_of) - dt.timedelta(days=1)).isoformat()


Splits = dict[str, list[tuple[str, float]]]


def load_splits(conn: sqlite3.Connection) -> Splits:
    """Share-count ratios of every split per ticker, sorted by ex-date: ticker -> [(ex_date, ratio)].

    Union of ``price_corporate_actions`` and the vendor action ledger kept in
    ``price_series_revisions.evidence_json``. Both tables are required: a missing
    one would read as "no splits", so it raises LookupError. The action table alone misses splits
    the price loader did not need to apply (NKTR 1-for-15 in 2025, NBR 1-for-50 in 2020),
    yet the stored closes are adjusted for them. Ratios above 1 add shares (a 2-for-1 is 2.0).
    """
    for table in ("price_corporate_actions", "price_series_revisions"):
        if not _has_table(conn, table):
            raise LookupError(f"database has no {table} table; split ratios cannot be established")
    found: dict[str, set[tuple[str, float]]] = {}
    for t, d, v in conn.execute(
            "SELECT ticker, action_date, value FROM price_corporate_actions WHERE action_type IN (?,?)",
            SPLIT_TYPES):
        found.setdefault(t, set()).add((d, round(float(v), SPLIT_ROUND)))
    for t, raw in conn.execute("SELECT ticker, evidence_json FROM price_series_revisions"):
        for a in json.loads(raw)["vendor_action_ledger"]:  # KeyError when a revision has no ledger: not guessed
            if a["action_type"] in SPLIT_TYPES:
                found.setdefault(t, set()).add((a["action_date"], round(float(a["value"]), SPLIT_ROUND)))
    return {t: sorted(v) for t, v in found.items()}


@dataclass
class AsOfView:
    """What a signal may see on ``as_of``: data knowable at that day's close.

    ``close`` and ``open`` are total-return-adjusted frames (dates x tickers) cut
    at ``as_of`` inclusive; each value uses only corporate actions with ex-date
    up to its own date. ``conn`` is exposed for read-only queries, but SEC facts
    must go through ``fundamentals``/``ttm``: a fact filed on day d is usable
    from the next session after d (filings can land after the close), so the
    cutoff is filed_date < as_of.
    """

    as_of: str
    close: pd.DataFrame
    open: pd.DataFrame
    conn: sqlite3.Connection
    splits: Splits
    diag: Diagnostics

    def fundamentals(self, ticker: str, tag, kind: str | None = None) -> Fact | None:
        return fundamentals_as_of(self.conn, ticker, self.filing_cutoff, tag, kind)

    def ttm(self, ticker: str, tag, period_end_max: str | None = None) -> Fact | None:
        return ttm_as_of(self.conn, ticker, self.filing_cutoff, tag, period_end_max)

    @property
    def filing_cutoff(self) -> str:
        return filing_cutoff(self.as_of)


class Market:
    """All sessions, adjusted prices and point-in-time membership of one DB."""

    def __init__(self, adj_open: pd.DataFrame, adj_close: pd.DataFrame, conn: sqlite3.Connection, *,
                 flagged: pd.DataFrame, price_tickers: set[str], spinoffs: pd.DataFrame,
                 exclusions: pd.DataFrame, splits: Splits, diag: Diagnostics):
        """Every input is required: an empty frame or dict must be passed on purpose, not defaulted."""
        self.adj_open, self.adj_close, self.conn = adj_open, adj_close, conn
        self.splits = splits
        self.load_diag = diag  # substitutions made while building the price matrix (once per load)
        self.diag = Diagnostics()  # substitutions made while scoring; run_bench starts each run with a fresh one
        self.exclusions = exclusions
        self.sessions: list[str] = list(adj_close.index)
        self.flagged = flagged
        self.price_tickers = price_tickers
        # Forward-filled copies value held positions across data gaps only.
        self.open_ff, self.close_ff = adj_open.ffill(), adj_close.ffill()
        self.columns = {t: i for i, t in enumerate(adj_close.columns)}
        self._members: dict[str, set[str]] = {}
        # Exit session (close of the session before the ex-date) per held column; see spinoffs.py.
        self.exits: dict[int, list[int]] = {}
        self.exit_events: dict[tuple[int, int], dict] = {}
        for r in spinoffs.itertuples():
            x = int(np.searchsorted(self.sessions, r.ex_date, side="left"))
            if r.parent_ticker not in self.columns:
                diag.note("spinoffs_not_applied", (r.parent_ticker, r.ex_date, "parent has no prices"))
            elif not 0 < x < len(self.sessions):
                diag.note("spinoffs_not_applied", (r.parent_ticker, r.ex_date, "ex-date outside the loaded sessions"))
            else:
                col = self.columns[r.parent_ticker]
                self.exits.setdefault(col, []).append(x - 1)
                self.exit_events[(col, x - 1)] = {"ticker": r.parent_ticker, "ex_date": r.ex_date,
                                                  "exit_date": self.sessions[x - 1], "spun_off": r.spun_off,
                                                  "kind": r.kind}
        for lst in self.exits.values():
            lst.sort()

    def exit_between(self, col: int, lo: int, hi: int) -> int | None:
        """First spin-off exit session q with lo <= q <= hi for column ``col``, else None."""
        return next((q for q in self.exits.get(col, ()) if lo <= q <= hi), None)

    def flagged_in(self, lo: int, hi: int) -> pd.DataFrame:
        """Flagged price rows dated inside sessions[lo..hi]; later rows are not counted."""
        a, b = self.sessions[lo], self.sessions[hi]
        f = self.flagged
        return f[(f["date"] >= a) & (f["date"] <= b)]

    def exclusion_summary(self, lo: int, hi: int) -> list[dict]:
        """Per (ticker, rule): index-member days inside sessions[lo..hi] removed by the validity filter."""
        a, b = self.sessions[lo], self.sessions[hi]
        ex = self.exclusions
        ex = ex[(ex["date"] >= a) & (ex["date"] <= b) & ex["date"].isin(self.sessions)]
        if ex.empty:
            return []
        mem = pd.read_sql_query("SELECT ticker, start_date, end_date FROM sp500_membership", self.conn)
        j = ex.merge(mem, on="ticker")
        j = j[(j["start_date"] <= j["date"]) & (j["end_date"].isna() | (j["end_date"] > j["date"]))]
        j = j.drop_duplicates(["ticker", "date", "rule"])
        g = j.groupby(["ticker", "rule"]).agg(member_days=("date", "size"), first=("date", "min"),
                                              last=("date", "max")).reset_index()
        return g.sort_values(["rule", "ticker"]).to_dict("records")

    def view(self, as_of: str) -> AsOfView:
        return AsOfView(as_of, self.adj_close.loc[:as_of], self.adj_open.loc[:as_of], self.conn, self.splits,
                        self.diag)

    def universe(self, idx: int) -> list[str]:
        """Point-in-time members on session ``idx`` that have a close that day."""
        d = self.sessions[idx]
        if d not in self._members:
            self._members[d] = members_on(self.conn, d)
        row = self.adj_close.iloc[idx]
        return sorted(t for t in self._members[d] if t in self.columns and np.isfinite(row[t]))

    def members_by_year(self, lo: int, hi: int) -> list[dict]:
        """Per calendar year in sessions[lo..hi]: distinct index members that year and how many have any close."""
        mem = pd.read_sql_query("SELECT ticker, start_date, end_date FROM sp500_membership", self.conn)
        years = np.array([d[:YEAR_CHARS] for d in self.sessions[lo:hi + 1]])
        out = []
        for y in sorted(set(years)):
            idx = np.nonzero(years == y)[0] + lo
            a, b = self.sessions[idx[0]], self.sessions[idx[-1]]
            names = set(mem.loc[(mem["start_date"] <= b) & (mem["end_date"].isna() | (mem["end_date"] > a)), "ticker"])
            window = self.adj_close.iloc[idx[0]:idx[-1] + 1]
            priced = {t for t in names if t in self.columns and window[t].notna().any()}
            out.append({"year": int(y), "members": len(names), "with_prices": len(priced)})
        return out

    def members_without_prices(self, lo: int, hi: int) -> tuple[int, int]:
        """(members in the window with no price rows at all, members in the window)."""
        a, b = self.sessions[lo], self.sessions[hi]
        rows = self.conn.execute(
            "SELECT DISTINCT ticker FROM sp500_membership "
            "WHERE start_date <= ? AND (end_date IS NULL OR end_date > ?)", (b, a)).fetchall()
        names = {r[0] for r in rows}
        return len(names - self.price_tickers), len(names)


def load_market(db_path: str | Path, spinoffs_path: str | Path | None = None) -> Market:
    diag = Diagnostics()
    conn = connect_ro(db_path)
    if not _has_table(conn, "sp500_membership"):
        conn.close()
        raise LookupError("database has no sp500_membership table; the bench needs point-in-time membership")
    prices, actions = load_total_return_inputs(db_path)
    tr = total_return_series(prices, actions)
    diag.count("dividends_after_last_price_not_applied", len(tr.attrs["dividends_after_last_price"]))
    for action_type, n in tr.attrs["ignored_action_types"].items():
        diag.count(f"corporate_actions_of_unknown_type_ignored:{action_type}", n)
    if not any(r[1] == "volume" for r in conn.execute("PRAGMA table_info(price_daily)")):
        conn.close()
        raise LookupError("price_daily has no volume column; the data-quality filter needs it")
    opens = pd.read_sql_query(
        "SELECT ticker, date, open FROM price_daily "
        "WHERE COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only' "
        "AND ticker NOT LIKE '^%'", conn)
    # Keep the real move (plus dividend) on every flagged row except a move on a split date,
    # where the close is probably not split-adjusted; see the module doc for why not 0%.
    prev = tr.groupby("ticker")["close"].shift(1)
    real = (tr["close"] + tr["dividend"]) / prev - 1.0
    keep_real = tr["flag"].isin(KEEP_REAL_MOVE_FLAGS) & (prev > 0) & (tr["close"] > 0)
    ret = tr["total_return"].where(~keep_real, real)
    # A listed spin-off whose raw drop is still in the data is not a loss: 0% on that day.
    spin = load_spinoffs(spinoffs_path) if spinoffs_path else load_spinoffs()
    neutral = {(r.parent_ticker, r.ex_date) for r in spin.itertuples() if r.price_effect == NEUTRALISED}
    is_neutral = pd.Series([(t, d) in neutral for t, d in zip(tr["ticker"], tr["date"])], index=tr.index)
    ret = ret.where(~is_neutral, 0.0)
    # A NaN return (flagged bad close or split-date move, or the day after an invalid close) is a
    # 0% day: the price level carries over. Counted so it shows in the report.
    nan_ret = ret.isna() & (tr.groupby("ticker").cumcount() > 0)
    diag.count("nan_returns_filled_as_0pct_on_flagged_rows", int((nan_ret & (tr["flag"] != "")).sum()))
    diag.count("nan_returns_filled_as_0pct_after_invalid_close", int((nan_ret & (tr["flag"] == "")).sum()))
    tr = tr.assign(growth=1.0 + ret.fillna(0.0))
    tr["adj_close"] = tr.groupby("ticker")["growth"].cumprod()
    tr.loc[~(tr["close"] > 0), "adj_close"] = np.nan
    flagged = tr.loc[tr["flag"] != "", ["ticker", "date", "flag"]].assign(
        move=real[tr["flag"] != ""],
        treatment=np.where(is_neutral[tr["flag"] != ""], "listed spin-off, 0% (spinoffs.csv)",
                           np.where(keep_real[tr["flag"] != ""], "real move kept", "0% day or NaN")))
    off_calendar = flagged["flag"] == "dividend_off_calendar"
    diag.count("off_calendar_dividend_flags_not_listed_in_flagged_rows", int(off_calendar.sum()))
    flagged = flagged[~off_calendar].reset_index(drop=True)
    adj_close = tr.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    raw_close = tr.pivot(index="date", columns="ticker", values="close").sort_index()
    raw_open = opens.pivot(index="date", columns="ticker", values="open").reindex(
        index=adj_close.index, columns=adj_close.columns)
    diag.count("nonpositive_opens_set_to_missing", int((raw_open <= 0).sum().sum()))
    raw_open = raw_open.where(raw_open > 0)
    adj_open = adj_close * (raw_open / raw_close)
    # Validity filter (quality.py): flagged ticker-days have no price, for every candidate and benchmark alike.
    volumes = pd.read_sql_query(
        "SELECT ticker, date, volume FROM price_daily "
        "WHERE COALESCE(adjustment_basis,'split_adjusted_price_only')='split_adjusted_price_only' "
        "AND ticker NOT LIKE '^%'", conn)
    junction = quality.junction_date(conn)
    if junction is None:
        diag.note("quality_rules_not_applied", "R2 seam: no unversioned price rows, so no backfill junction to check")
    bad = quality.invalid_rows(tr[["ticker", "date", "close"]].merge(volumes, on=["ticker", "date"], how="left"),
                               junction)
    for t, d in bad[["ticker", "date"]].drop_duplicates().itertuples(index=False):
        if t not in adj_close.columns or d not in adj_close.index:  # cannot happen: the rows came from this frame
            raise KeyError(f"invalid row {t} {d} is not in the price matrix")
        adj_close.loc[d, t] = np.nan
        adj_open.loc[d, t] = np.nan
    if BENCHMARK_TICKER not in adj_close.columns:
        conn.close()
        raise LookupError(f"{BENCHMARK_TICKER} prices are required for the benchmark")
    calendar = adj_close.index[adj_close[BENCHMARK_TICKER].notna()]
    diag.count("dates_dropped_for_no_benchmark_close", len(adj_close.index) - len(calendar))
    in_calendar = flagged["date"].isin(calendar)
    diag.count("flagged_rows_on_dates_without_benchmark_close_not_listed", int((~in_calendar).sum()))
    return Market(adj_open.loc[calendar], adj_close.loc[calendar], conn,
                  flagged=flagged[in_calendar], price_tickers=set(tr["ticker"].unique()), spinoffs=spin,
                  exclusions=bad, splits=load_splits(conn), diag=diag)
