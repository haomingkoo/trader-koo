from __future__ import annotations

from typing import Iterable

import pandas as pd


CANONICAL_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
REQUIRED_PRICE_COLUMNS = ["date", "open", "high", "low", "close"]
ALIASES = {
    "datetime": "date",
    "timestamp": "date",
    "vol": "volume",
}


def _norm_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    cols: list[str] = []
    for col in df.columns:
        if not isinstance(col, tuple):
            cols.append(_norm_name(col))
            continue
        parts = [_norm_name(x) for x in col if str(x).strip() and str(x).strip().lower() != "none"]
        if not parts:
            cols.append("")
            continue
        chosen = parts[0]
        for p in parts:
            p2 = ALIASES.get(p, p)
            if p2 in CANONICAL_COLUMNS:
                chosen = p2
                break
        cols.append(ALIASES.get(chosen, chosen))
    out = df.copy()
    out.columns = cols
    return out


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [ALIASES.get(_norm_name(c), _norm_name(c)) for c in out.columns]
    return out


def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    out = pd.DataFrame(index=df.index)
    for col in pd.Index(df.columns).unique():
        block = df.loc[:, df.columns == col]
        if isinstance(block, pd.Series):
            out[col] = block
            continue
        if block.shape[1] == 1:
            out[col] = block.iloc[:, 0]
            continue
        # Prefer first non-null value left-to-right for duplicate aliases.
        out[col] = block.bfill(axis=1).iloc[:, 0]
    return out


def ensure_ohlcv_schema(
    df: pd.DataFrame,
    *,
    date_column: str | None = None,
    allow_empty: bool = True,
) -> pd.DataFrame:
    if df is None:
        if allow_empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        raise ValueError("Input dataframe is None")

    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    work = _flatten_columns(df)
    if date_column and date_column in work.columns and date_column != "date":
        work = work.rename(columns={date_column: "date"})

    work = _coerce_columns(work)
    if "close" not in work.columns:
        if "adj_close" in work.columns:
            work = work.rename(columns={"adj_close": "close"})
        elif "adjclose" in work.columns:
            work = work.rename(columns={"adjclose": "close"})

    if "date" not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={work.columns[0]: "date"})
        elif work.index.name and _norm_name(work.index.name) in {"date", "datetime", "timestamp"}:
            work = work.reset_index().rename(columns={work.columns[0]: "date"})

    if "date" not in work.columns:
        if allow_empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        raise ValueError("Unable to locate date column")

    work = _collapse_duplicate_columns(work)

    for c in REQUIRED_PRICE_COLUMNS:
        if c not in work.columns:
            if allow_empty:
                return pd.DataFrame(columns=CANONICAL_COLUMNS)
            raise ValueError(f"Missing required column: {c}")

    keep = [c for c in CANONICAL_COLUMNS if c in work.columns]
    work = work[keep].copy()
    if "volume" not in work.columns:
        work["volume"] = pd.NA

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = (
        work.dropna(subset=["date", "open", "high", "low", "close"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return work[CANONICAL_COLUMNS].copy()


def pick_columns(rows: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = rows.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    return out[list(columns)]
