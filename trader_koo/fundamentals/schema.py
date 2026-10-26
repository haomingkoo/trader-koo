"""Additive, idempotent schema for SEC point-in-time fundamentals."""
from __future__ import annotations

import sqlite3

SEC_SCHEMA = """
CREATE TABLE IF NOT EXISTS sec_fundamentals (
    cik INTEGER NOT NULL,
    taxonomy TEXT NOT NULL,
    tag TEXT NOT NULL,
    unit TEXT NOT NULL,
    period_start TEXT NOT NULL DEFAULT '',
    period_end TEXT NOT NULL,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    value REAL NOT NULL,
    form TEXT,
    accession TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    frame TEXT,
    PRIMARY KEY (cik, tag, unit, period_start, period_end, accession)
);
CREATE INDEX IF NOT EXISTS idx_sec_fundamentals_cik_tag_filed
    ON sec_fundamentals (cik, tag, filed_date);
CREATE TABLE IF NOT EXISTS sec_cik_map (
    ticker TEXT NOT NULL,
    cik INTEGER NOT NULL,
    title TEXT,
    mapped_ts TEXT NOT NULL,
    PRIMARY KEY (ticker, cik)
);
CREATE TABLE IF NOT EXISTS sec_fetch_status (
    cik INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    http_status INTEGER,
    n_facts INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT,
    fetched_ts TEXT NOT NULL,
    error TEXT
);
"""


def ensure_sec_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SEC_SCHEMA)
    conn.commit()
