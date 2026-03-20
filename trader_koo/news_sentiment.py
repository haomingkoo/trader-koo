"""External market-news sentiment via Finnhub.

Uses two Finnhub free-tier endpoints:
- ``/api/v1/news-sentiment?symbol=X`` — pre-computed bullish/bearish %
- ``/api/v1/company-news?symbol=X&from=…&to=…`` — recent headlines

Falls back to Alpha Vantage NEWS_SENTIMENT if ``TRADER_KOO_ALPHA_VANTAGE_KEY``
is set and ``FINNHUB_API_KEY`` is not.

Results are cached in-process to stay within provider rate limits.
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
_DEFAULT_LOOKBACK_HOURS = 72
_DEFAULT_LIMIT = 50
_DEFAULT_CACHE_TTL_SEC = 900

_cache_lock = threading.Lock()
_cache_expires_at: dt.datetime | None = None
_cache_payload: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _finnhub_key() -> str:
    return str(os.getenv("FINNHUB_API_KEY", "")).strip()


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
    raw = str(os.getenv("TRADER_KOO_NEWS_TIMEOUT_SEC", "20")).strip()
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


# ---------------------------------------------------------------------------
# Scoring helpers (shared by both providers)
# ---------------------------------------------------------------------------

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
    provider: str,
    now_utc: dt.datetime,
    tickers: list[str],
    lookback_hours: int,
    note: str,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "source_type": "news",
        "available": False,
        "score": None,
        "raw_score": None,
        "label": None,
        "article_count": 0,
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": lookback_hours,
        "tickers": tickers,
        "topics": [],
        "note": note,
        "headlines": [],
    }


# ---------------------------------------------------------------------------
# Finnhub provider
# ---------------------------------------------------------------------------

def _finnhub_get(path: str, params: dict[str, str], api_key: str) -> Any:
    """Make a GET request to Finnhub and return parsed JSON."""
    params["token"] = api_key
    qs = urllib.parse.urlencode(params)
    url = f"https://finnhub.io/api/v1/{path}?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trader-koo/1.0 (+https://trader.kooexperience.com)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_request_timeout_sec()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_finnhub_news_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    tickers = _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)
    lookback_hours = _lookback_hours()
    api_key = _finnhub_key()

    if not api_key:
        return _empty_news_payload(
            provider="finnhub",
            now_utc=now_utc,
            tickers=tickers,
            lookback_hours=lookback_hours,
            note=(
                "Finnhub API key not configured. "
                "Set FINNHUB_API_KEY in Railway env vars to enable news sentiment."
            ),
        )

    # Step 1: Fetch company news per ticker and score headlines with lexicon.
    # Note: /news-sentiment is premium-only (403 on free tier).
    # /company-news is free and returns recent headlines we can score ourselves.
    from trader_koo.rss_news import _score_headline

    date_to = now_utc.strftime("%Y-%m-%d")
    date_from = (now_utc - dt.timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")

    weighted_total = 0.0
    weight_total = 0.0
    ticker_details: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            articles = _finnhub_get(
                "company-news",
                {"symbol": ticker, "from": date_from, "to": date_to},
                api_key,
            )
            if not isinstance(articles, list):
                continue

            ticker_bullish = 0
            ticker_bearish = 0
            ticker_scores: list[float] = []

            for item in articles[:30]:  # cap per ticker to save processing
                if not isinstance(item, dict):
                    continue
                headline = str(item.get("headline") or "").strip()
                summary = str(item.get("summary") or "").strip()
                if not headline:
                    continue
                raw, bullish_count, bearish_count = _score_headline(headline, summary)
                if raw is not None and raw != 0.0:
                    ticker_scores.append(raw)
                    ticker_bullish += bullish_count
                    ticker_bearish += bearish_count

            if ticker_scores:
                avg_raw = sum(ticker_scores) / len(ticker_scores)
                weighted_total += avg_raw * len(ticker_scores)
                weight_total += len(ticker_scores)
                ticker_details.append({
                    "ticker": ticker,
                    "bullish": ticker_bullish,
                    "bearish": ticker_bearish,
                    "articles_scored": len(ticker_scores),
                    "articles_total": min(len(articles), 30),
                    "avg_raw_score": round(avg_raw, 4),
                })
            else:
                ticker_details.append({
                    "ticker": ticker,
                    "bullish": 0,
                    "bearish": 0,
                    "articles_scored": 0,
                    "articles_total": min(len(articles), 30),
                    "avg_raw_score": None,
                })
        except Exception as exc:
            LOG.warning("Finnhub company-news failed for %s: %s", ticker, exc)

    # Step 2: Fetch headlines via RSS (scored) with Finnhub company-news fallback
    headlines: list[dict[str, Any]] = []
    rss_meta: dict[str, Any] | None = None
    try:
        from trader_koo.rss_news import fetch_rss_headlines

        rss_result = fetch_rss_headlines(tickers=tickers, max_headlines=10, now_utc=now_utc)
        if rss_result.get("available") and rss_result.get("headlines"):
            headlines = rss_result["headlines"][:10]
            rss_meta = {
                "rss_article_count": rss_result.get("article_count", 0),
                "rss_feed_breakdown": rss_result.get("feed_breakdown", []),
            }
    except Exception as exc:
        LOG.warning("RSS headline fetch failed, falling back to Finnhub company-news: %s", exc)

    # Fallback to Finnhub company-news if RSS returned nothing
    if not headlines:
        date_to = now_utc.strftime("%Y-%m-%d")
        date_from = (now_utc - dt.timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
        for ticker in tickers[:2]:
            try:
                articles = _finnhub_get(
                    "company-news",
                    {"symbol": ticker, "from": date_from, "to": date_to},
                    api_key,
                )
                if not isinstance(articles, list):
                    continue
                for item in articles:
                    if len(headlines) >= 5:
                        break
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("headline") or "").strip()
                    if not title:
                        continue
                    ts = item.get("datetime")
                    time_published = None
                    if isinstance(ts, int | float):
                        time_published = _iso_utc(
                            dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
                        )
                    headlines.append({
                        "title": title,
                        "source": str(item.get("source") or "").strip() or None,
                        "url": str(item.get("url") or "").strip() or None,
                        "time_published": time_published,
                        "score": None,
                        "label": None,
                    })
            except Exception as exc:
                LOG.warning("Finnhub company-news failed for %s: %s", ticker, exc)

    if weight_total <= 0:
        return _empty_news_payload(
            provider="finnhub",
            now_utc=now_utc,
            tickers=tickers,
            lookback_hours=lookback_hours,
            note="Finnhub returned no usable sentiment scores for tracked tickers",
        )

    raw_score = weighted_total / weight_total
    score = _score_to_100(raw_score)
    article_count = sum(d.get("articles_scored", d.get("articles_total", 0)) for d in ticker_details)

    return {
        "provider": "finnhub",
        "source_type": "news",
        "available": True,
        "score": score,
        "raw_score": round(raw_score, 4),
        "label": _label_for_score(score),
        "article_count": article_count,
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": lookback_hours,
        "tickers": tickers,
        "topics": [],
        "note": (
            f"Finnhub news sentiment aggregated across {len(ticker_details)} tickers "
            f"({article_count} articles in last week)"
            + (f" + {rss_meta['rss_article_count']} RSS headlines" if rss_meta else "")
        ),
        "headlines": headlines,
        "ticker_details": ticker_details,
        **(rss_meta or {}),
    }


# ---------------------------------------------------------------------------
# Alpha Vantage provider (legacy fallback)
# ---------------------------------------------------------------------------

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


def _fetch_alpha_vantage_news_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    tickers = _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)
    topics = _csv_env("TRADER_KOO_SENTIMENT_TOPICS", (
        "financial_markets", "economy_macro", "economy_monetary", "earnings",
    ))
    lookback_hours = _lookback_hours()
    api_key = _alpha_vantage_key()

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
            provider="alpha_vantage",
            now_utc=now_utc,
            tickers=tickers,
            lookback_hours=lookback_hours,
            note=f"Alpha Vantage request failed: {exc}",
        )

    feed = payload.get("feed")
    if not isinstance(feed, list) or not feed:
        note = str(
            payload.get("Information") or payload.get("Note")
            or "No news sentiment articles returned"
        ).strip()
        return _empty_news_payload(
            provider="alpha_vantage",
            now_utc=now_utc,
            tickers=tickers,
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
            provider="alpha_vantage",
            now_utc=now_utc,
            tickers=tickers,
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


# ---------------------------------------------------------------------------
# RSS-only fallback (no API key required)
# ---------------------------------------------------------------------------

def _fetch_rss_only_news_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    """Use RSS feeds as a standalone news sentiment source when no API keys are configured."""
    tickers = _csv_env("TRADER_KOO_SENTIMENT_TICKERS", _DEFAULT_TICKERS)
    lookback_hours = _lookback_hours()

    try:
        from trader_koo.rss_news import fetch_rss_headlines

        rss = fetch_rss_headlines(tickers=tickers, max_headlines=10, now_utc=now_utc)
        if rss.get("available") and rss.get("score") is not None:
            return {
                "provider": "rss_aggregator",
                "source_type": "news",
                "available": True,
                "score": rss["score"],
                "raw_score": rss.get("raw_score"),
                "label": rss.get("label"),
                "article_count": rss.get("article_count", 0),
                "updated_at": _iso_utc(now_utc),
                "lookback_hours": lookback_hours,
                "tickers": tickers,
                "topics": [],
                "note": (
                    f"RSS-only mode (no API key configured). {rss.get('note', '')}. "
                    "Set FINNHUB_API_KEY for more accurate sentiment scoring."
                ),
                "headlines": rss.get("headlines", []),
                "rss_feed_breakdown": rss.get("feed_breakdown", []),
            }
    except Exception as exc:
        LOG.warning("RSS-only news sentiment failed: %s", exc)

    return _empty_news_payload(
        provider="rss_aggregator",
        now_utc=now_utc,
        tickers=tickers,
        lookback_hours=lookback_hours,
        note=(
            "No news sentiment provider configured and RSS feeds unavailable. "
            "Set FINNHUB_API_KEY (recommended, free 60 calls/min) for news sentiment."
        ),
    )


# ---------------------------------------------------------------------------
# Public API — dispatches to Finnhub (primary) or Alpha Vantage (fallback)
# ---------------------------------------------------------------------------

def get_external_news_sentiment(*, now_utc: dt.datetime | None = None, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached external news sentiment metadata.

    Provider priority:
    1. Finnhub (if ``FINNHUB_API_KEY`` is set)
    2. Alpha Vantage (if ``TRADER_KOO_ALPHA_VANTAGE_KEY`` is set)
    3. Empty payload with configuration note
    """
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

    if _finnhub_key():
        fresh_payload = _fetch_finnhub_news_sentiment(resolved_now)
    elif _alpha_vantage_key():
        fresh_payload = _fetch_alpha_vantage_news_sentiment(resolved_now)
    else:
        # No API key — fall back to RSS-only headlines with lexicon scoring
        fresh_payload = _fetch_rss_only_news_sentiment(resolved_now)

    expires_at = resolved_now + dt.timedelta(seconds=_cache_ttl_sec())

    with _cache_lock:
        _cache_payload = copy.deepcopy(fresh_payload)
        _cache_expires_at = expires_at

    return copy.deepcopy(fresh_payload)
