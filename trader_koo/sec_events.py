"""Recent SEC 8-K filings for the top report setups (display only).

EDGAR's submissions feed is free, needs no key, and stamps every filing with
the time SEC accepted it, so a filing is shown only if it was accepted before
the report's cutoff. Nothing here feeds a score, a ranking or a paper-trade
input: the result lives under its own report key, not on the setup rows.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any
from zoneinfo import ZoneInfo

import requests

from trader_koo.config import env_bool, env_int
from trader_koo.fundamentals import config as sec_cfg
from trader_koo.fundamentals.loader import sec_ticker

LOG = logging.getLogger("trader_koo.sec_events")

ENABLED = env_bool("TRADER_KOO_SETUP_SEC_EVENTS_ENABLED", True)
MAX_TICKERS = env_int("TRADER_KOO_SETUP_SEC_EVENTS_MAX_TICKERS", 12, min_value=0, max_value=40)
LOOKBACK_DAYS = 30
MAX_PER_TICKER = 5
HTTP_TIMEOUT_SEC = 15
REQUEST_GAP_SEC = 0.25  # 4 requests/second, under SEC's 10/second fair-access limit
CIK_MAP_TTL_SEC = 24 * 3600
MAX_FAILURES = 3  # then skip the rest, so an EDGAR outage costs seconds, not minutes
FORMS = frozenset({"8-K", "8-K/A"})
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
_ET = ZoneInfo("America/New_York")

# 8-K item numbers in plain words. 9.01 (exhibits) only says documents are
# attached, so it is dropped unless it is the filing's only item.
ITEM_LABELS: dict[str, str] = {
    "1.01": "Signed a material agreement",
    "1.02": "Ended a material agreement",
    "1.03": "Bankruptcy or receivership",
    "1.04": "Mine safety violation",
    "1.05": "Cybersecurity incident",
    "2.01": "Completed an acquisition or sale",
    "2.02": "Earnings results",
    "2.03": "Took on new debt",
    "2.04": "Debt repayment triggered early",
    "2.05": "Restructuring or exit costs",
    "2.06": "Asset write-down",
    "3.01": "Listing standard notice",
    "3.02": "Unregistered share sale",
    "3.03": "Change to shareholder rights",
    "4.01": "Auditor change",
    "4.02": "Past financials no longer reliable",
    "5.01": "Change in control",
    "5.02": "Executive or director change",
    "5.03": "Bylaws or fiscal year changed",
    "5.04": "Pension plan trading halt",
    "5.05": "Code of ethics change",
    "5.07": "Shareholder vote results",
    "5.08": "Director nomination notice",
    "7.01": "Investor update (Reg FD)",
    "8.01": "Other company event",
    "9.01": "Exhibits filed",
}

GetJson = Callable[[str], Any]

_cik_cache: dict[str, Any] = {"loaded_at": 0.0, "map": {}}


def _http_get_json(url: str) -> Any:
    resp = requests.get(
        url,
        headers={"User-Agent": sec_cfg.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
        timeout=HTTP_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    time.sleep(REQUEST_GAP_SEC)
    return resp.json()


def _cik_map(get_json: GetJson, *, now: float) -> dict[str, int]:
    if _cik_cache["map"] and now - _cik_cache["loaded_at"] < CIK_MAP_TTL_SEC:
        return _cik_cache["map"]
    payload = get_json(sec_cfg.SEC_TICKERS_URL)
    mapping = {
        sec_ticker(str(v["ticker"])): int(v["cik_str"])
        for v in (payload or {}).values()
        if isinstance(v, dict) and v.get("ticker") and v.get("cik_str") is not None
    }
    if not mapping:
        raise ValueError("SEC ticker map was empty")
    _cik_cache.update(loaded_at=now, map=mapping)
    return mapping


def cutoff_for(as_of_date: str | None, now_utc: dt.datetime) -> dt.datetime:
    """End of the report's market date in New York time, never later than now."""
    if not as_of_date:
        return now_utc
    day = dt.datetime.fromisoformat(str(as_of_date)).date()
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), tzinfo=_ET)
    return min(end.astimezone(dt.timezone.utc), now_utc)


def _accepted_at(value: Any) -> dt.datetime | None:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _labels(items: str) -> list[str]:
    codes = [code.strip() for code in str(items or "").split(",") if code.strip()]
    if not codes:
        return ["Filing (no item listed)"]
    labels = [ITEM_LABELS.get(code, f"Item {code}") for code in codes if code != "9.01"]
    return labels or [ITEM_LABELS["9.01"]]


def filings_from_submissions(
    payload: dict[str, Any], *, cik: int, cutoff: dt.datetime,
    lookback_days: int = LOOKBACK_DAYS, limit: int = MAX_PER_TICKER,
) -> list[dict[str, Any]]:
    """8-K filings accepted in (cutoff - lookback, cutoff], newest first."""
    recent = ((payload or {}).get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    start = cutoff - dt.timedelta(days=lookback_days)
    out: list[dict[str, Any]] = []
    for i, form in enumerate(forms):
        if form not in FORMS:
            continue
        accepted = _accepted_at(recent["acceptanceDateTime"][i])
        if accepted is None or not (start < accepted <= cutoff):
            continue
        accession = str(recent["accessionNumber"][i])
        out.append({
            "form": form,
            "filed_date": str(recent["filingDate"][i]),
            "accepted_at": accepted.astimezone(dt.timezone.utc).isoformat(),
            "items": _labels(recent["items"][i]),
            "url": FILING_URL.format(
                cik=cik, accession=accession.replace("-", ""), document=recent["primaryDocument"][i],
            ),
        })
    out.sort(key=lambda f: f["accepted_at"], reverse=True)
    return out[:limit]


def build_setup_sec_events(
    tickers: Iterable[str],
    *,
    as_of_date: str | None,
    now_utc: dt.datetime | None = None,
    get_json: GetJson | None = None,
    enabled: bool | None = None,
    max_tickers: int | None = None,
) -> dict[str, Any]:
    """Report surface: {available, cutoff, tickers: {T: [filings]}, unmapped, errors}."""
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    get_json = get_json or _http_get_json
    enabled = ENABLED if enabled is None else enabled
    limit = MAX_TICKERS if max_tickers is None else max_tickers
    surface: dict[str, Any] = {
        "enabled": enabled, "available": False, "source": "SEC EDGAR",
        "lookback_days": LOOKBACK_DAYS, "tickers": {}, "unmapped": [], "errors": {},
    }
    if not enabled:
        return surface
    chosen: list[str] = []
    for raw in tickers:
        ticker = str(raw or "").upper().strip()
        if ticker and not ticker.startswith("^") and ticker not in chosen:
            chosen.append(ticker)
        if len(chosen) >= limit:
            break
    try:
        cutoff = cutoff_for(as_of_date, now_utc)
    except ValueError:
        surface["error"] = "invalid_as_of_date"
        LOG.warning("SEC events skipped: invalid as_of_date %r", as_of_date)
        return surface
    surface["cutoff"] = cutoff.isoformat()
    if not chosen:
        return surface
    try:
        cik_map = _cik_map(get_json, now=time.monotonic())
    except Exception as exc:  # any failure: report it, show nothing
        surface["error"] = f"cik_map_failed: {type(exc).__name__}"
        LOG.warning("SEC events unavailable, ticker map failed: %s", exc)
        return surface
    for ticker in chosen:
        if len(surface["errors"]) >= MAX_FAILURES:
            surface["errors"][ticker] = "skipped_after_failures"
            continue
        cik = cik_map.get(sec_ticker(ticker))
        if cik is None:
            surface["unmapped"].append(ticker)
            continue
        try:
            payload = get_json(SUBMISSIONS_URL.format(cik=cik))
            surface["tickers"][ticker] = filings_from_submissions(payload, cik=cik, cutoff=cutoff)
        except Exception as exc:  # one ticker failing must not hide the others
            surface["errors"][ticker] = type(exc).__name__
            LOG.warning("SEC events for %s failed: %s", ticker, exc)
    surface["available"] = bool(surface["tickers"])
    return surface
