"""Data for hypotheses v3: SPY, QQQ and ^VIX daily bars aligned on SPY's sessions, cut before the sealed window.

The bars are read straight from ``price_daily`` with ``date < boundary``; nothing dated on or after the
pinned seal is selected, so the sealed window is never opened. Stored rows are
``split_adjusted_price_only`` (no dividends in the price): the bench's raw index-ETF closes, not a
total-return series. Every row that is dropped or left empty is counted in ``diag``.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import connect_ro
from .diagnostics import Diagnostics

ETFS = ("SPY", "QQQ")
VIX = "^VIX"
FIELDS = ("open", "high", "low", "close", "volume")
BASIS = "split_adjusted_price_only"
FOMC_FILE = Path(__file__).resolve().parents[3] / "docs" / "research" / "fomc-decision-dates-2010-2023.csv"
FOMC_FIRST_YEAR, FOMC_LAST_YEAR = 2010, 2023  # hypotheses-v3 H10: "2010 to 2023"
FOMC_KIND = "scheduled"


@dataclass
class Frame:
    """Aligned arrays over the SPY sessions of the window (index i is session ``dates[i]``)."""

    dates: list[str]
    bars: dict[str, dict[str, np.ndarray]]  # ticker -> field -> (N,) float
    ex_dividend: dict[str, np.ndarray]      # ETF -> (N,) bool: session is an ex-dividend date
    fomc: np.ndarray | None                 # (N,) bool: scheduled FOMC decision day; None when the list is incomplete
    fomc_missing: list[int]                 # years of 2010-2023 with no scheduled meeting in the list
    diag: Diagnostics
    basis: dict

    @property
    def n(self) -> int:
        return len(self.dates)

    def weekday(self) -> np.ndarray:
        return pd.to_datetime(pd.Index(self.dates)).dayofweek.to_numpy()

    def month(self) -> np.ndarray:
        return pd.to_datetime(pd.Index(self.dates)).month.to_numpy() - 1


def load_fomc(path: Path = FOMC_FILE) -> tuple[set[str], list[int]]:
    """Scheduled decision days from the sourced CSV, and the years 2010-2023 for which none is listed."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    dates = {r["date"] for r in rows if r["kind"] == FOMC_KIND}
    years = {d[:4] for d in dates}
    missing = [y for y in range(FOMC_FIRST_YEAR, FOMC_LAST_YEAR + 1) if str(y) not in years]
    return dates, missing


def load_frame(db_path: str | Path, boundary: str, fomc_path: Path = FOMC_FILE) -> Frame:
    diag = Diagnostics()
    conn = connect_ro(db_path)
    try:
        rows = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume, COALESCE(adjustment_basis,?) AS basis, "
            "basis_status FROM price_daily WHERE ticker IN (?,?,?) AND date < ?",
            conn, params=(BASIS, *ETFS, VIX, boundary))
        div = pd.read_sql_query(
            "SELECT ticker, action_date FROM price_corporate_actions "
            "WHERE action_type='dividend' AND ticker IN (?,?) AND action_date < ?", conn, params=(*ETFS, boundary))
    finally:
        conn.close()
    other = rows[rows["basis"] != BASIS]
    diag.count("rows_with_another_adjustment_basis_dropped", len(other))
    basis = {"adjustment_basis": rows["basis"].value_counts().to_dict(),
             "basis_status": rows["basis_status"].fillna("NULL").value_counts().to_dict()}
    rows = rows[rows["basis"] == BASIS]
    sessions = sorted(rows.loc[rows["ticker"] == "SPY", "date"])
    if not sessions:
        raise LookupError("no SPY rows before the seal; the calendar of v3 is SPY's sessions")
    idx = pd.Index(sessions)
    bars: dict[str, dict[str, np.ndarray]] = {}
    for t in (*ETFS, VIX):
        g = rows[rows["ticker"] == t].set_index("date")
        diag.count(f"{t}_rows_on_dates_without_a_SPY_session_dropped", int((~g.index.isin(idx)).sum()))
        g = g.reindex(idx)
        diag.count(f"{t}_sessions_without_a_row", int(g["close"].isna().sum()))
        bars[t] = {f: g[f].to_numpy(float) for f in FIELDS}
        for f in FIELDS:  # a bar that is not positive is a missing bar, counted
            bad = bars[t][f] < 0 if f == "volume" else bars[t][f] <= 0
            diag.count(f"{t}_{f}_nonpositive_set_to_missing", int(bad.sum()))
            bars[t][f] = np.where(bad, np.nan, bars[t][f])
    ex = {}
    for t in ETFS:
        d = set(div.loc[div["ticker"] == t, "action_date"])
        ex[t] = idx.isin(d)
        diag.count(f"{t}_dividend_dates_not_on_a_session", len(d) - int(ex[t].sum()))
    fomc_dates, missing = load_fomc(fomc_path)
    if missing:
        for y in missing:
            diag.note("fomc_years_without_a_scheduled_meeting", str(y))
    fomc = None if missing else idx.isin(fomc_dates)
    return Frame(sessions, bars, ex, fomc, missing, diag, basis)
