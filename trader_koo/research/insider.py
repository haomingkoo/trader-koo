"""SEC insider trades (Forms 4 and 4/A, open-market buys and sells) from the quarterly Form 3/4/5 data sets.

Source: SEC "Insider Transactions Data Sets", one zip per calendar quarter of filing dates,
2006 Q1 onward. Each trade row keeps its filing date, which is the date the trade became
public; a signal may use it from the next session (the same cutoff as SEC facts).

Only non-derivative transactions with code P (open-market purchase) or S (open-market sale)
are stored: awards, option exercises, gifts and tax withholding say little about an insider's
view. Every quarter loaded is recorded with the sha256 of its zip, so a load is reproducible.

Usage: python -m trader_koo.research.insider --db <research db> --cache <dir> [--first 2006q1] [--last 2026q2]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import sqlite3
import sys
import zipfile
from pathlib import Path

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.sec_client import SecClient

# The SEC publishes these paths; they are part of the source definition, not tunables.
DATASET_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{quarter}_form345.zip"
FIRST_QUARTER = "2006q1"
FORMS = frozenset({"4", "4/A"})
TRADE_CODES = frozenset({"P", "S"})
QUARTERS_PER_YEAR = 4
MONTHS_PER_QUARTER = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sec_insider_trades (
    accession TEXT NOT NULL,
    trans_sk INTEGER NOT NULL,
    quarter TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    document_type TEXT NOT NULL,
    issuer_cik INTEGER NOT NULL,
    issuer_symbol TEXT,
    owner_ciks TEXT NOT NULL,
    is_director INTEGER NOT NULL,
    is_officer INTEGER NOT NULL,
    is_ten_pct_owner INTEGER NOT NULL,
    officer_titles TEXT,
    trans_date TEXT,
    trans_code TEXT NOT NULL,
    shares REAL,
    price REAL,
    acquired_disposed TEXT,
    shares_after REAL,
    direct_indirect TEXT,
    PRIMARY KEY (accession, trans_sk)
);
CREATE INDEX IF NOT EXISTS idx_sec_insider_issuer_filed ON sec_insider_trades (issuer_cik, filing_date);
CREATE TABLE IF NOT EXISTS sec_insider_quarters (
    quarter TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    zip_sha256 TEXT,
    trade_rows INTEGER,
    loaded_ts TEXT NOT NULL
);
"""

REQUIRED = {
    "SUBMISSION.tsv": ("ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"),
    "REPORTINGOWNER.tsv": ("ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"),
    "NONDERIV_TRANS.tsv": ("ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                           "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
                           "DIRECT_INDIRECT_OWNERSHIP"),
}


class DatasetFormatError(ValueError):
    """A quarterly file is missing a table or column this loader depends on."""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def quarters(first: str, last: str) -> list[str]:
    """Calendar quarters from ``first`` to ``last`` inclusive, as ``2015q1``."""
    def parse(q: str) -> tuple[int, int]:
        year, n = q.lower().split("q")
        return int(year), int(n)
    (y, n), end = parse(first), parse(last)
    out = []
    while (y, n) <= end:
        out.append(f"{y}q{n}")
        y, n = (y + 1, 1) if n == QUARTERS_PER_YEAR else (y, n + 1)
    return out


def last_complete_quarter(today: dt.date) -> str:
    q = (today.month - 1) // MONTHS_PER_QUARTER + 1
    return f"{today.year - 1}q{QUARTERS_PER_YEAR}" if q == 1 else f"{today.year}q{q - 1}"


def _iso(day: str) -> str | None:
    """SEC writes dates as 31-MAR-2015."""
    return dt.datetime.strptime(day, "%d-%b-%Y").date().isoformat() if day else None


def _num(text: str) -> float | None:
    return float(text) if text else None


def _table(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    if name not in archive.namelist():
        raise DatasetFormatError(f"{name} missing from the archive")
    with archive.open(name) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"),
                                delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in REQUIRED[name] if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetFormatError(f"{name} lacks columns {missing}")
        return list(reader)


def parse_trades(payload: bytes, quarter: str) -> list[tuple]:
    """Rows for ``sec_insider_trades`` from one quarterly zip."""
    archive = zipfile.ZipFile(io.BytesIO(payload))
    subs = {r["ACCESSION_NUMBER"]: r for r in _table(archive, "SUBMISSION.tsv") if r["DOCUMENT_TYPE"] in FORMS}
    owners: dict[str, list[dict[str, str]]] = {}
    for r in _table(archive, "REPORTINGOWNER.tsv"):
        if r["ACCESSION_NUMBER"] in subs:
            owners.setdefault(r["ACCESSION_NUMBER"], []).append(r)
    rows = []
    for t in _table(archive, "NONDERIV_TRANS.tsv"):
        acc = t["ACCESSION_NUMBER"]
        if acc not in subs or t["TRANS_CODE"] not in TRADE_CODES:
            continue
        sub, own = subs[acc], owners.get(acc, [])
        rel = ",".join(o["RPTOWNER_RELATIONSHIP"] for o in own)
        titles = "; ".join(sorted({o["RPTOWNER_TITLE"] for o in own if o["RPTOWNER_TITLE"]})) or None
        rows.append((
            acc, int(t["NONDERIV_TRANS_SK"]), quarter, _iso(sub["FILING_DATE"]), sub["DOCUMENT_TYPE"],
            int(sub["ISSUERCIK"]), sub["ISSUERTRADINGSYMBOL"].strip().upper() or None,
            ",".join(sorted({o["RPTOWNERCIK"].lstrip("0") for o in own})),
            int("Director" in rel), int("Officer" in rel), int("TenPercentOwner" in rel), titles,
            _iso(t["TRANS_DATE"]), t["TRANS_CODE"], _num(t["TRANS_SHARES"]), _num(t["TRANS_PRICEPERSHARE"]),
            t["TRANS_ACQUIRED_DISP_CD"] or None, _num(t["SHRS_OWND_FOLWNG_TRANS"]),
            t["DIRECT_INDIRECT_OWNERSHIP"] or None,
        ))
    return rows


def load_quarter(conn: sqlite3.Connection, quarter: str, payload: bytes) -> int:
    rows = parse_trades(payload, quarter)
    with conn:
        conn.execute("DELETE FROM sec_insider_trades WHERE quarter = ?", (quarter,))
        if rows:
            conn.executemany(f"INSERT INTO sec_insider_trades VALUES ({','.join('?' * len(rows[0]))})", rows)
        conn.execute("INSERT OR REPLACE INTO sec_insider_quarters VALUES (?, 'loaded', ?, ?, ?)",
                     (quarter, hashlib.sha256(payload).hexdigest(), len(rows),
                      dt.datetime.now(dt.timezone.utc).isoformat()))
    return len(rows)


def fetch(client: SecClient, cache: Path, quarter: str) -> bytes | None:
    """The quarter's zip from the cache, else from the SEC; None when the SEC has not published it."""
    path = cache / f"{quarter}_form345.zip"
    if path.exists():
        return path.read_bytes()
    status, body = client.get_bytes(DATASET_URL.format(quarter=quarter))
    if status == cfg.HTTP_NOT_FOUND or body is None:
        return None
    path.write_bytes(body)
    return body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trader_koo.research.insider")
    p.add_argument("--db", required=True)
    p.add_argument("--cache", required=True, help="directory for the downloaded quarterly zips")
    p.add_argument("--first", default=FIRST_QUARTER)
    p.add_argument("--last", default=last_complete_quarter(dt.datetime.now(dt.timezone.utc).date()))
    p.add_argument("--reload", action="store_true", help="re-parse quarters already loaded")
    args = p.parse_args(argv)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    done = {q for (q,) in conn.execute("SELECT quarter FROM sec_insider_quarters WHERE status = 'loaded'")}
    client = SecClient()
    for q in quarters(args.first, args.last):
        if q in done and not args.reload:
            continue
        payload = fetch(client, cache, q)
        if payload is None:
            print(f"{q}: not published by the SEC", file=sys.stderr)
            with conn:
                conn.execute("INSERT OR REPLACE INTO sec_insider_quarters VALUES (?, 'not_published', NULL, NULL, ?)",
                             (q, dt.datetime.now(dt.timezone.utc).isoformat()))
            continue
        print(f"{q}: {load_quarter(conn, q, payload)} open-market trades")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
