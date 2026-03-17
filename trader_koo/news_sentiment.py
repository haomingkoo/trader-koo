"""Optional external market-news sentiment sources.

The current implementation uses Alpha Vantage's NEWS_SENTIMENT endpoint when
``TRADER_KOO_ALPHA_VANTAGE_KEY`` is configured. Results are cached in-process
to avoid burning through provider rate limits.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from typing import Any

LOG = logging.getLogger(__name__)

_DEFAULT_TICKERS = ("SPY", "QQQ", "DIA", "IWM")
_DEFAULT_TOPICS = (
    "financial_markets",
    "economy_macro",
    "economy_monetary",
    "earnings",
)
_DEFAULT_LOOKBACK_HOURS = 72
_DEFAULT_LIMIT = 50
_DEFAULT_CACHE_TTL_SEC = 900

_cache_lock = threading.Lock()
_cache_expires_at: dt.datetime | None = None
_cache_payload: dict[str, Any] | None = None


def _alpha_vantage_key() -> str:
    return str(
        os.getenv("TRADER_KOO_ALPHA_VANTAGE_KEY", "")
        or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    ).strip()


def _cache_ttl_sec() -> int:
    raw = str(os.getenv("TRADER_KOO_SENTIMENT_CACHE_TTL_SEC", _DEFAULT_CACHE_TTL_SEC)).strip()
    try:
        return max(60, int(raw))
    except ValueError:
        return _DEFAULT_CACHE_TTL_SEC


def _request_timeout_sec() -> int:
    raw = str(os.getenv("TRADER_KOO_ALPHA_VANTAGE_TIMEOUT_SEC", "20")).strip()
    try:
        return max(5, int(raw))
    except ValueError:
        return 20


def _lookback_hours() -> int:
    raw = str(os.getenv("TRADER_KOO_SENTIMENT_LOOKBACK_HOURS", _DEFAULT_LOOKBACK_HOURS)).strip()
    try:
        return max(12, min(168, int(raw)))
    except ValueError:
        return _DEFAULT_LOOKBACK_HOURS


def _result_limit() -> int:
    raw = str(os.getenv("TRADER_KOO_SENTIMENT_LIMIT", _DEFAULT_LIMIT)).strip()
    try:
        return max(10, min(100, int(raw)))
    except ValueError:
        return _DEFAULT_LIMIT


def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return list(default)
    return [
        token
        for token in {
            str(part or "").strip()
            for part in raw.split(",")
        }
        if token
    ]


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_to_100(score: float | None) -> int | None:
    if score is None:
        return None
    clipped = max(-1.0, min(1.0, score))
    return round((clipped + 1.0) * 50.0)


def _label_for_score(score: int | None) -> str | None:
    if score is None:
        return None
    if score < 25:
        return "Extreme Fear"
    if score < 45:
        return "Fear"
    if score < 55:
        return "Neutral"
    if score < 75:
        return "Greed"
    return "Extreme Greed"


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _empty_news_payload(
    *,
    now_utc: dt.datetime,
    tickers: list[str],
    topics: list[str],
    lookback_hours: int,
    note: str,
) -> dict[str, Any]:
    return {
        "provider": "alpha_vantage",
        "source_type": "news",
        "available": False,
        "score": None,
        "raw_score": None,
        "label": None,
        "article_count": 0,
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": lookback_hours,
        "tickers": tickers,
        "topics": topics,
        "note": note,
        "headlines": [],
    }


def _article_score(item: dict[str, Any], tracked_tickers: set[str]) -> tuple[float | None, float]:
    ticker_sentiment = item.get("ticker_sentiment")
    if isinstance(ticker_sentiment, list):
        weighted_total = 0.0
        weight_total = 0.0
        for entry in ticker_sentiment:
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker") or "").strip().upper()
            if ticker and ticker not in tracked_tickers:
                continue
            score = _safe_float(entry.get("ticker_sentiment_score"))
            relevance = _safe_float(entry.get("relevance_score"))
            if score is None:
                continue
            weight = max(relevance or 0.0, 0.05)
            weighted_total += score * weight
            weight_total += weight
        if weight_total > 0:
            return weighted_total / weight_total, weight_total

    score = _safe_float(item.get("overall_sentiment_score"))
    if score is None:
        return None, 0.0
    return score, 1.0


def _headline_entry(item: dict[str, Any], tracked_tickers: set[str]) -> dict[str, Any]:
    raw_score, _ = _article_score(item, tracked_tickers)
    score = _score_to_100(raw_score)
    return {
        "title": str(item.get("title") or "").strip(),
        "source": str(item.get("source") or "").strip() or None,
        "url": str(item.get("url") or "").strip() or None,
        "time_published": str(item.get("time_published") or "").strip() or None,
        "score": score,
        "label": _label_for_score(score),
    }


def _summarize_alpha_vantage_feed(
    payload: dict[str, Any],
    *,
    now_utc: dt.datetime,
    tickers: list[str],
    topics: list[str],
    lookback_hours: int,
) -> dict[str, Any]:
    feed = payload.get("feed")
    if not isinstance(feed, list) or not feed:
        note = str(payload.get("Information") or payload.get("Note") or "No news sentiment articles returned").strip()
        return _empty_news_payload(
            now_utc=now_utc,
            tickers=tickers,
            topics=topics,
            lookback_hours=lookback_hours,
            note=note,
        )

    tracked_tickers = {ticker.upper() for ticker in tickers}
    weighted_total = 0.0
    weight_total = 0.0
    headlines: list[dict[str, Any]] = []

    for raw_item in feed:
        if not isinstance(raw_item, dict):
            continue
        score, weight = _article_score(raw_item, tracked_tickers)
        if score is not None and weight > 0:
            weighted_total += score * weight
            weight_total += weight
        if len(headlines) < 5:
            entry = _headline_entry(raw_item, tracked_tickers)
            if entry["title"]:
                headlines.append(entry)

    if weight_total <= 0:
        return _empty_news_payload(
            now_utc=now_utc,
            tickers=tickers,
            topics=topics,
            lookback_hours=lookback_hours,
            note="Alpha Vantage returned articles without usable sentiment scores",
        )

    raw_score = weighted_total / weight_total
    score = _score_to_100(raw_score)
    article_count = sum(1 for item in feed if isinstance(item, dict))

    return {
        "provider": "alpha_vantage",
        "source_type": "news",
        "available": True,
        "score": score,
        "raw_score": round(raw_score, 4),
        "label": _label_for_score(score),
        "article_count": article_count,
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": lookback_hours,
        "tickers": tickers,
        "topics": topics,
        "note": f"{article_count} Alpha Vantage articles aggregated over the last {lookback_hours}h",
        "headlines": headlines,
    }


def _fetch_alpha_vantage_news_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    tickers = _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)
    topics = _csv_env("TRADER_KOO_SENTIMENT_TOPICS", _DEFAULT_TOPICS)
    lookback_hours = _lookback_hours()
    api_key = _alpha_vantage_key()

    if not api_key:
        return _empty_news_payload(
            now_utc=now_utc,
            tickers=tickers,
            topics=topics,
            lookback_hours=lookback_hours,
            note="Configure TRADER_KOO_ALPHA_VANTAGE_KEY to enable external news sentiment.",
        )

    time_from = (now_utc - dt.timedelta(hours=lookback_hours)).strftime("%Y%m%dT%H%M")
    qs = urllib.parse.urlencode(
        {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "topics": ",".join(topics),
            "time_from": time_from,
            "sort": "LATEST",
            "limit": str(_result_limit()),
            "apikey": api_key,
        }
    )
    url = f"https://www.alphavantage.co/query?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trader-koo/1.0 (+https://trader.kooexperience.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=_request_timeout_sec()) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
    except Exception as exc:
        LOG.warning("Alpha Vantage news sentiment fetch failed: %s", exc)
        return _empty_news_payload(
            now_utc=now_utc,
            tickers=tickers,
            topics=topics,
            lookback_hours=lookback_hours,
            note=f"Alpha Vantage request failed: {exc}",
        )

    return _summarize_alpha_vantage_feed(
        payload,
        now_utc=now_utc,
        tickers=tickers,
        topics=topics,
        lookback_hours=lookback_hours,
    )


def get_external_news_sentiment(*, now_utc: dt.datetime | None = None, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached external news sentiment metadata."""
    global _cache_expires_at, _cache_payload

    resolved_now = now_utc or dt.datetime.now(dt.timezone.utc)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=dt.timezone.utc)
    else:
        resolved_now = resolved_now.astimezone(dt.timezone.utc)

    with _cache_lock:
        if (
            not force_refresh
            and _cache_payload is not None
            and _cache_expires_at is not None
            and resolved_now < _cache_expires_at
        ):
            return copy.deepcopy(_cache_payload)

    fresh_payload = _fetch_alpha_vantage_news_sentiment(resolved_now)
    expires_at = resolved_now + dt.timedelta(seconds=_cache_ttl_sec())

    with _cache_lock:
        _cache_payload = copy.deepcopy(fresh_payload)
        _cache_expires_at = expires_at

    return copy.deepcopy(fresh_payload)
