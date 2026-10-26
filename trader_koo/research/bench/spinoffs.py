"""Spin-off ex-dates of index-member parents, loaded from the sourced ``spinoffs.csv``.

Why: on the ex-date the parent's close falls by the value of the spun-off shares
while holders receive those shares, so a raw close-to-close return shows a loss
that never happened (CTVA, -84% on 2026-10-01). Yahoo usually repairs this
afterwards, either as a split-style factor on history or as a dividend, but not
always at once, and the repair is not point-in-time. The bench therefore never
holds a parent across a listed ex-date: it sells at the close of the session
before and does not buy a name whose ex-date falls inside the coming holding
period. Spin-offs are announced weeks ahead, so the list is knowable in advance.

Columns: parent_ticker (price_daily symbol); ex_date (first session whose regular-way
close excludes the spun-off shares); spun_off; kind (``spinoff`` or ``backfill_seam``, a
date where the stored price basis jumps because backfilled rows were fetched after the
vendor adjusted for a spin-off while the seeded rows were not); date_basis (how the
source gives the date); price_effect (``vendor_adjusted``: history was repaired and the
move is continuous in the research DB; ``raw_drop``: the raw drop is present and the
return on that day is set to 0% in the chain); source_url (company filing, press release
or exchange/OCC notice). Only events with a source are listed. Every ``spinoff`` row was
also checked against the research DB: the vendor stored a split-style or dividend action
on exactly the ex-date, except CTVA, whose repair had not happened yet.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).with_name("spinoffs.csv")
COLUMNS = ["parent_ticker", "ex_date", "spun_off", "kind", "date_basis", "price_effect", "source_url"]
KINDS = {"spinoff", "backfill_seam"}
PRICE_EFFECTS = {"vendor_adjusted", "raw_drop"}
NEUTRALISED = "raw_drop"


def load_spinoffs(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns must be {COLUMNS}, got {list(frame.columns)}")
    for row in frame.itertuples(index=False):
        dt.date.fromisoformat(row.ex_date)
        if row.kind not in KINDS or row.price_effect not in PRICE_EFFECTS:
            raise ValueError(f"{path}: bad kind or price_effect for {row.parent_ticker} {row.ex_date}")
        if not row.source_url.startswith("https://"):
            raise ValueError(f"{path}: {row.parent_ticker} {row.ex_date} has no https source")
    if frame.duplicated(["parent_ticker", "ex_date"]).any():
        raise ValueError(f"{path}: duplicate (parent_ticker, ex_date)")
    return frame.sort_values(["ex_date", "parent_ticker"]).reset_index(drop=True)
