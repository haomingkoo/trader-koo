"""SEC point-in-time fundamentals. Fixtures are trimmed real EDGAR responses."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from trader_koo.fundamentals import config as cfg
from trader_koo.fundamentals.loader import (
    add_predecessor_cik,
    fact_rows,
    fetch_cik,
    load_universe,
    sec_ticker,
    store_cik_map,
    universe_tickers,
)
from trader_koo.fundamentals.pit import fundamentals_as_of, ttm_as_of
from trader_koo.fundamentals.schema import ensure_sec_schema
from trader_koo.fundamentals.sec_client import SecClient, SecFetchError

FIX = Path(__file__).parent / "fixtures" / "sec"
AAPL_CIK = 320193
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _load(name: str):
    return json.loads((FIX / name).read_text())


class FakeResp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, responses: dict[str, list]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.headers: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        return self.responses[url].pop(0)


def _client(responses, sleeps=None):
    clock = {"t": 0.0}

    def sleep(s):
        if sleeps is not None:
            sleeps.append(s)
        clock["t"] += s

    return SecClient(FakeSession(responses), sleep=sleep, clock=lambda: clock["t"])


def _url(cik: int) -> str:
    return cfg.SEC_FACTS_URL.format(cik=cik)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    ensure_sec_schema(c)
    store_cik_map(c, _load("company_tickers.json"))
    return c


@pytest.fixture()
def loaded(conn):
    client = _client({_url(AAPL_CIK): [FakeResp(200, _load("companyfacts_aapl.json"))]})
    assert fetch_cik(conn, client, AAPL_CIK).status == "ok"
    return conn


# --- schema ---------------------------------------------------------------

def test_schema_is_idempotent_and_additive(conn):
    before = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    ensure_sec_schema(conn)
    ensure_sec_schema(conn)
    after = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert before == after
    assert not [n for n in after if n.startswith(("paper_", "report_"))]


# --- client ---------------------------------------------------------------

def test_client_declares_user_agent_and_stays_under_rate_limit():
    sleeps: list[float] = []
    client = _client({"u": [FakeResp(200, {}) for _ in range(5)]}, sleeps)
    assert client.session.headers["User-Agent"] == "trader-koo research haomingkoo@gmail.com"
    for _ in range(5):
        client.get_json("u")
    # clock only advances by sleeping, so total sleep == 4 intervals of 1/rate
    assert sum(sleeps) == pytest.approx(4 / cfg.REQUESTS_PER_SECOND)
    assert cfg.REQUESTS_PER_SECOND < 10


def test_client_backs_off_then_succeeds():
    sleeps: list[float] = []
    client = _client({"u": [FakeResp(429), FakeResp(503), FakeResp(200, {"ok": 1})]}, sleeps)
    assert client.get_json("u") == (200, {"ok": 1})
    assert max(sleeps) >= cfg.BACKOFF_BASE_SEC * 2


def test_client_gives_up_and_404_is_not_an_error():
    client = _client({"u": [FakeResp(500)] * cfg.MAX_ATTEMPTS, "v": [FakeResp(404)]})
    with pytest.raises(SecFetchError):
        client.get_json("u")
    assert client.get_json("v") == (404, None)


# --- loader ---------------------------------------------------------------

def test_loader_stores_only_listed_tags_with_raw_values(loaded):
    tags = {r[0] for r in loaded.execute("SELECT DISTINCT tag FROM sec_fundamentals")}
    assert tags <= {t for _, t in cfg.STORED_TAGS}
    assert REV in tags and "EntityCommonStockSharesOutstanding" in tags
    row = loaded.execute(
        "SELECT value, filed_date, accession FROM sec_fundamentals WHERE tag=? AND period_end='2023-09-30'"
        " AND period_start='2022-09-25' ORDER BY filed_date",
        (REV,),
    ).fetchall()
    assert row[0] == (383285000000.0, "2023-11-03", "0000320193-23-000106")
    assert len(row) == 3  # same value re-reported in three later 10-Ks, kept as separate accessions


def test_loader_is_idempotent_and_records_coverage(loaded):
    n = loaded.execute("SELECT COUNT(*) FROM sec_fundamentals").fetchone()[0]
    client = _client({_url(AAPL_CIK): [FakeResp(200, _load("companyfacts_aapl.json"))]})
    fetch_cik(loaded, client, AAPL_CIK)
    assert loaded.execute("SELECT COUNT(*) FROM sec_fundamentals").fetchone()[0] == n
    status, n_facts, tags = loaded.execute("SELECT status, n_facts, tags_json FROM sec_fetch_status").fetchone()
    assert (status, n_facts) == ("ok", n)
    assert REV in json.loads(tags)


def test_missing_tags_are_recorded_not_fabricated(conn):
    client = _client({_url(789019): [FakeResp(200, _load("companyfacts_msft.json"))]})
    fetch_cik(conn, client, 789019)
    tags = set(json.loads(conn.execute("SELECT tags_json FROM sec_fetch_status").fetchone()[0]))
    all_tags = {t for _, t in cfg.STORED_TAGS}
    missing = all_tags - tags
    assert missing  # fixture lacks some listed tags; no rows exist for them
    assert not conn.execute(
        f"SELECT 1 FROM sec_fundamentals WHERE tag IN ({','.join('?' * len(missing))})", tuple(missing)
    ).fetchone()


def test_load_universe_resumes_and_dedupes_shared_cik(conn):
    # GOOG and GOOGL share one CIK; BRK-B has an SEC-spelled ticker.
    googl = conn.execute("SELECT cik FROM sec_cik_map WHERE ticker='GOOGL'").fetchone()[0]
    assert googl == conn.execute("SELECT cik FROM sec_cik_map WHERE ticker='GOOG'").fetchone()[0]
    client = _client({_url(AAPL_CIK): [FakeResp(200, _load("companyfacts_aapl.json"))], _url(googl): [FakeResp(404)]})
    counts = load_universe(conn, client, ["AAPL", "GOOG", "GOOGL", "ZZZZ"])
    assert counts == {"ok": 1, "empty": 0, "no_facts": 1, "error": 0, "skipped": 1, "unmapped": 1,
                      "duplicate_facts_ignored": counts["duplicate_facts_ignored"]}
    again = _client({})
    counts = load_universe(conn, again, ["AAPL", "GOOGL"])
    assert counts["skipped"] == 2 and again.session.calls == []


def test_failed_cik_is_retried_on_next_run(conn):
    client = _client({_url(AAPL_CIK): [FakeResp(500)] * cfg.MAX_ATTEMPTS})
    assert load_universe(conn, client, ["AAPL"])["error"] == 1
    retry = _client({_url(AAPL_CIK): [FakeResp(200, _load("companyfacts_aapl.json"))]})
    assert load_universe(conn, retry, ["AAPL"])["ok"] == 1


def test_ticker_spelling_and_universe(tmp_path):
    assert sec_ticker("brk.b") == "BRK-B"
    u = sqlite3.connect(tmp_path / "u.db")
    u.execute("CREATE TABLE finviz_fundamentals (ticker TEXT)")
    u.executemany("INSERT INTO finviz_fundamentals VALUES (?)", [("AAPL",), ("AAPL",), ("BRK-B",), ("^VIX",)])
    with pytest.raises(LookupError):  # departed members would silently be missing
        universe_tickers(u)
    u.execute("CREATE TABLE sp500_membership (ticker TEXT)")
    u.execute("INSERT INTO sp500_membership VALUES ('FB')")
    assert universe_tickers(u) == ["AAPL", "BRK-B", "FB"]


def test_fact_rows_match_payload_counts():
    payload = _load("companyfacts_aapl.json")
    expected = sum(
        len(v)
        for tax, tag in cfg.STORED_TAGS
        for v in payload["facts"].get(tax, {}).get(tag, {}).get("units", {}).values()
    )
    assert len(list(fact_rows(AAPL_CIK, payload))) == expected


# --- point in time --------------------------------------------------------

def test_value_not_visible_before_filed_date(loaded):
    filed = "2023-11-03"
    before = fundamentals_as_of(loaded, "AAPL", "2023-11-02", REV, kind="annual")
    on = fundamentals_as_of(loaded, "AAPL", filed, REV, kind="annual")
    assert before.period_end == "2022-09-24" and before.filed_date < filed
    assert (on.period_end, on.value) == ("2023-09-30", 383285000000.0)


def test_lookahead_guard_over_every_filing_date(loaded):
    dates = [r[0] for r in loaded.execute("SELECT DISTINCT filed_date FROM sec_fundamentals ORDER BY 1")]
    for d in dates:
        for concept in cfg.CONCEPTS:
            fact = fundamentals_as_of(loaded, "AAPL", d, concept)
            assert fact is None or fact.filed_date <= d
            ttm = ttm_as_of(loaded, "AAPL", d, concept)
            assert ttm is None or ttm.filed_date <= d
    assert fundamentals_as_of(loaded, "AAPL", "1999-01-01", REV) is None


def test_restatement_appears_only_from_its_own_filed_date(loaded):
    def ni(as_of):
        return fundamentals_as_of(loaded, "AAPL", as_of, "NetIncomeLoss", kind="annual")

    assert ni("2009-10-26") is None  # nothing filed yet
    assert (ni("2009-10-27").period_end, ni("2009-10-27").value) == ("2008-09-27", 4834000000.0)
    assert ni("2010-01-24").value == 4834000000.0  # 10-K/A not yet filed
    assert ni("2010-01-25").value == 6119000000.0  # restated by the 10-K/A
    assert ni("2010-01-25").form == "10-K/A"


def test_concept_fallback_uses_newest_period_across_tags(loaded):
    # SalesRevenueNet ended in 2018; the ASC 606 tag carries recent periods.
    fact = fundamentals_as_of(loaded, "AAPL", "2026-08-01", "revenue", kind="annual")
    assert fact.tag == REV and fact.period_end == "2025-09-27"


def test_ttm_follows_the_ytd_rule_and_never_peeks(loaded):
    after_10k = ttm_as_of(loaded, "AAPL", "2025-11-01", REV)
    assert (after_10k.period_end, after_10k.value) == ("2025-09-27", 416161000000.0)
    # newest filing is the Q2 10-Q (filed 2026-05-01): FY25 + 6M FY26 - 6M FY25
    q2 = ttm_as_of(loaded, "AAPL", "2026-07-30", REV)
    assert q2.value == 416161000000.0 + 254940000000.0 - 219659000000.0
    assert q2.period_end == "2026-03-28" and q2.filed_date == "2026-05-01"
    q3 = ttm_as_of(loaded, "AAPL", "2026-08-01", REV)
    assert q3.value == 416161000000.0 + 364357000000.0 - 313695000000.0
    assert q3.period_end == "2026-06-27" and q3.filed_date == "2026-07-31"


def test_predecessor_cik_is_loaded_and_read_under_the_same_ticker(conn):
    # SEC's map points a ticker at its newest registrant only; a reorganisation
    # leaves the earlier history under the predecessor CIK.
    add_predecessor_cik(conn, "AAPL", 789019)
    client = _client({
        _url(AAPL_CIK): [FakeResp(200, _load("companyfacts_aapl.json"))],
        _url(789019): [FakeResp(200, _load("companyfacts_msft.json"))],
    })
    assert load_universe(conn, client, ["AAPL"])["ok"] == 2
    ciks = {r[0] for r in conn.execute("SELECT DISTINCT cik FROM sec_fundamentals")}
    assert ciks == {AAPL_CIK, 789019}
    assert fundamentals_as_of(conn, "AAPL", "2026-08-01", "NetIncomeLoss", kind="annual") is not None


def test_ttm_fails_closed_when_a_component_is_missing(loaded):
    # No complete prior fiscal year exists in the data at this date.
    assert ttm_as_of(loaded, "AAPL", "2022-07-01", "NetIncomeLoss") is None
    assert ttm_as_of(loaded, "AAPL", "2026-08-01", "CommonStockSharesOutstanding") is None


# --- no hidden fallbacks ---------------------------------------------------

def test_payload_without_facts_is_an_error_not_an_empty_success(conn):
    client = _client({_url(AAPL_CIK): [FakeResp(200, {"cik": AAPL_CIK})]})
    assert fetch_cik(conn, client, AAPL_CIK).status == "error"
    row = conn.execute("SELECT status, error FROM sec_fetch_status WHERE cik=?", (AAPL_CIK,)).fetchone()
    assert row[0] == "error" and "no 'facts'" in row[1]


def test_company_with_no_stored_tags_is_counted_as_empty(conn):
    client = _client({_url(AAPL_CIK): [FakeResp(200, {"facts": {"us-gaap": {}}})]})
    counts = load_universe(conn, client, ["AAPL"])
    assert counts["empty"] == 1 and counts["ok"] == 0


def test_duplicate_facts_are_counted_when_a_cik_is_refreshed(conn):
    payload = _load("companyfacts_aapl.json")
    client = _client({_url(AAPL_CIK): [FakeResp(200, payload), FakeResp(200, payload)]})
    first = load_universe(conn, client, ["AAPL"])
    again = load_universe(conn, client, ["AAPL"], refresh=True)
    assert first["duplicate_facts_ignored"] == 0
    assert again["duplicate_facts_ignored"] == conn.execute("SELECT COUNT(*) FROM sec_fundamentals").fetchone()[0]


def test_coverage_report_has_no_invented_zero_without_data(conn):
    from trader_koo.fundamentals.report import coverage_report

    u = sqlite3.connect(":memory:")
    u.execute("CREATE TABLE finviz_fundamentals (ticker TEXT, snapshot_ts TEXT, raw_json TEXT)")
    u.execute("INSERT INTO finviz_fundamentals VALUES ('AAPL', '2026-01-01', 'not json')")
    out = coverage_report(conn, u)
    assert out["restated_pct"] is None and out["period_groups_with_restated_values"] is None
    assert out["tickers_with_unreadable_sector_json"] == ["AAPL"]
