"""Rule-based validity filter for stored price rows (data-quality rules, version 1).

Fixed before any hypothesis result was looked at, and applied identically to
every candidate, reference run, equal-weight benchmark and random control: a
flagged ticker-day has no price in the bench, so the ticker is not in the
universe that day and cannot be scored, bought or used in a benchmark.

The rules look at the stored series only, never at returns of a candidate. They
are not signals: a row that fails them is an artifact of the data source, so
using the following rows of a run to recognise it is not trading information.

The limits (5 sessions, 25%, 25%, 7 moves) are settings in config/research.toml; the text below
describes them as registered with rules version 1.

R1 zero_volume_run   5 or more consecutive stored rows with volume 0 or missing. The
                     whole run is invalid (the source filled prices without trading).
R2 seam              The stored series is the backfilled history joined to the series
                     stored before the backfill, starting at ``junction`` (the first
                     date of the originally stored rows). If the close on the first
                     row at or after the junction differs from the last close before
                     it by more than 25%, the backfilled spell does not sit on the
                     same price basis as the adjacent series and the whole backfilled
                     spell of that ticker (every row before the junction) is invalid.
R3 erratic_year      A ticker-calendar-year holding 7 or more close-to-close moves
                     above 25% in absolute value. The most any listed index member has
                     in a year in this database without being a source artifact is 6
                     (March 2020).
"""
from __future__ import annotations

import pandas as pd

from trader_koo.research.settings import get as _settings

_Q = _settings().quality
ZERO_VOLUME_RUN = _Q.zero_volume_run_sessions
SEAM_MOVE = _Q.seam_move_fraction
ERRATIC_MOVE = _Q.erratic_move_fraction
ERRATIC_MOVES_PER_YEAR = _Q.erratic_moves_per_year
YEAR_CHARS = 4  # ISO date prefix width
COLUMNS = ["ticker", "date", "rule"]
RULES = ("zero_volume_run", "seam", "erratic_year")


def _zero_volume_runs(g: pd.DataFrame) -> pd.DataFrame:
    zero = ~(g["volume"] > 0)
    run = (zero != zero.groupby(g["ticker"]).shift()).cumsum()
    size = zero.groupby(run).transform("size")
    return g.loc[zero & (size >= ZERO_VOLUME_RUN), ["ticker", "date"]].assign(rule="zero_volume_run")


def _seams(g: pd.DataFrame, junction: str | None) -> pd.DataFrame:
    if junction is None:
        return pd.DataFrame(columns=COLUMNS)
    out = []
    for ticker, px in g.groupby("ticker", sort=False):
        before, after = px[px["date"] < junction], px[px["date"] >= junction]
        if before.empty or after.empty:
            continue
        move = after["close"].iloc[0] / before["close"].iloc[-1] - 1.0
        if not abs(move) <= SEAM_MOVE:  # NaN counts as inconsistent
            out.append(before[["ticker", "date"]].assign(rule="seam"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)


def _erratic_years(g: pd.DataFrame) -> pd.DataFrame:
    move = g["close"] / g.groupby("ticker")["close"].shift() - 1.0
    year = g["date"].str[:YEAR_CHARS]
    big = (move.abs() > ERRATIC_MOVE).groupby([g["ticker"], year]).transform("sum")
    return g.loc[big >= ERRATIC_MOVES_PER_YEAR, ["ticker", "date"]].assign(rule="erratic_year")


def invalid_rows(prices: pd.DataFrame, junction: str | None) -> pd.DataFrame:
    """(ticker, date, rule) for every stored row that fails a rule; a row can fail several.

    ``prices`` has ticker, date (ISO text), close and volume.
    """
    g = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    parts = [_zero_volume_runs(g), _seams(g, junction), _erratic_years(g)]
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def junction_date(conn) -> str | None:
    """First date of the originally stored (unversioned) rows, else None."""
    row = conn.execute("SELECT MIN(date) FROM price_daily WHERE adjustment_basis IS NULL").fetchone()
    return row[0] if row and row[0] else None
