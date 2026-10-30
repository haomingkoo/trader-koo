"""SEC insider trades loader, on real rows: every 2015 Q1 Form 3/4/5 filing for AAPL, JPM and KO.

The fixture zip holds those issuers' rows of SUBMISSION, REPORTINGOWNER and NONDERIV_TRANS as the SEC
published them in 2015q1_form345.zip.
"""
from __future__ import annotations

import datetime as dt
import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from trader_koo.research import insider

FIX = Path(__file__).parent / "fixtures" / "sec_insider" / "2015q1_form345_sample.zip"


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    insider.ensure_schema(c)
    return c


def test_keeps_only_open_market_buys_and_sells_on_form_4(conn):
    n = insider.load_quarter(conn, "2015q1", FIX.read_bytes())
    # Hand count from the TSVs: AAPL 8 S; JPM 1 P + 11 S; KO 2 P + 4 S (KO's Form 5 gifts excluded).
    assert n == 26
    by = dict(conn.execute("SELECT issuer_symbol || trans_code, COUNT(*) FROM sec_insider_trades GROUP BY 1"))
    assert by == {"AAPLS": 8, "JPMP": 1, "JPMS": 11, "KOP": 2, "KOS": 4}


def test_purchase_row_matches_the_filing(conn):
    insider.load_quarter(conn, "2015q1", FIX.read_bytes())
    row = conn.execute(
        "SELECT filing_date, trans_date, issuer_cik, owner_ciks, is_director, is_officer, shares, price, shares_after "
        "FROM sec_insider_trades WHERE accession = '0001225208-15-001528'").fetchone()
    # Director Michael A. Neal bought 9,050 JPM at 55.07 on 2015-01-15, filed 2015-01-16.
    assert row == ("2015-01-16", "2015-01-15", 19617, "1233167", 1, 0, 9050.0, 55.07, 14525.46)


def test_reload_replaces_the_quarter_and_records_the_zip_hash(conn):
    payload = FIX.read_bytes()
    insider.load_quarter(conn, "2015q1", payload)
    insider.load_quarter(conn, "2015q1", payload)
    assert conn.execute("SELECT COUNT(*) FROM sec_insider_trades").fetchone()[0] == 26
    status, sha, rows = conn.execute("SELECT status, zip_sha256, trade_rows FROM sec_insider_quarters").fetchone()
    assert (status, rows, len(sha)) == ("loaded", 26, 64)


def test_missing_column_fails_closed():
    src = zipfile.ZipFile(FIX)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for name in src.namelist():
            text = src.read(name).decode()
            if name == "NONDERIV_TRANS.tsv":
                text = text.replace("TRANS_CODE", "TRANS_KIND", 1)
            out.writestr(name, text)
    with pytest.raises(insider.DatasetFormatError, match="TRANS_CODE"):
        insider.parse_trades(buf.getvalue(), "2015q1")


def test_quarter_range_and_last_published_quarter():
    assert insider.quarters("2025q3", "2026q2") == ["2025q3", "2025q4", "2026q1", "2026q2"]
    assert insider.last_complete_quarter(dt.date(2026, 10, 30)) == "2026q3"
    assert insider.last_complete_quarter(dt.date(2027, 2, 1)) == "2026q4"
