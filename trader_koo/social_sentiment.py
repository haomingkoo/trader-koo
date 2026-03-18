"""Social sentiment for the market dashboard.

Provider priority:
1. StockTwits public API (no auth, 200 req/hr, works from datacenter IPs)
2. Reddit public JSON (fallback, blocked from most cloud IPs)

StockTwits messages include user-tagged bullish/bearish sentiment labels,
giving us real retail trader sentiment without needing NLP.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import math
import os
import re
import threading
import urllib.parse
import urllib.request
from typing import Any

LOG = logging.getLogger(__name__)

_DEFAULT_TICKERS = ("SPY", "QQQ", "AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "META", "AMD", "GME")
_DEFAULT_CACHE_TTL_SEC = 900
_DEFAULT_POST_LIMIT = 30

_cache_lock = threading.Lock()
_cache_expires_at: dt.datetime | None = None
_cache_payload: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _csv_env(name: str, default: tuple[str, ...]) -> list[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return list(default)
    values = {
        str(part or "").strip()
        for part in raw.split(",")
        if str(part or "").strip()
    }
    return list(values) if values else list(default)


def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = str(os.getenv(name, default)).strip()
    try:
        return max(lo, min(hi, int(raw)))
    except ValueError:
        return default


def _request_timeout_sec() -> int:
    return _int_env("TRADER_KOO_SOCIAL_TIMEOUT_SEC", 15, lo=5, hi=60)


def _cache_ttl_sec() -> int:
    return _int_env("TRADER_KOO_SOCIAL_SENTIMENT_CACHE_TTL_SEC", _DEFAULT_CACHE_TTL_SEC, lo=60, hi=3600)


def _post_limit() -> int:
    return _int_env("TRADER_KOO_SOCIAL_POST_LIMIT", _DEFAULT_POST_LIMIT, lo=10, hi=50)


def _iso_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _score_to_100(raw_score: float | None) -> int | None:
    if raw_score is None:
        return None
    clipped = max(-1.0, min(1.0, raw_score))
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


def _empty_social_payload(
    *,
    provider: str,
    now_utc: dt.datetime,
    tickers: list[str],
    note: str,
    source_breakdown: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "source_type": "social",
        "available": False,
        "score": None,
        "raw_score": None,
        "label": None,
        "post_count": 0,
        "subreddit_count": 0,
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": 24,
        "subreddits": [],
        "tickers": tickers,
        "note": note,
        "bullish_terms_total": 0,
        "bearish_terms_total": 0,
        "posts": [],
        "source_breakdown": source_breakdown or [],
    }


# ---------------------------------------------------------------------------
# StockTwits provider
# ---------------------------------------------------------------------------

def _stocktwits_get(symbol: str) -> dict[str, Any]:
    """Fetch recent messages for a symbol from StockTwits public API."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{urllib.parse.quote(symbol)}.json"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "trader-koo/1.0 (+https://trader.kooexperience.com)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_request_timeout_sec()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_stocktwits_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    tickers = _csv_env("TRADER_KOO_SOCIAL_TICKERS", _DEFAULT_TICKERS)

    total_bullish = 0
    total_bearish = 0
    total_messages = 0
    posts: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []

    for ticker in tickers:
        try:
            data = _stocktwits_get(ticker)
            messages = data.get("messages") or []
            if not isinstance(messages, list):
                messages = []

            ticker_bullish = 0
            ticker_bearish = 0
            ticker_count = 0

            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                entities = msg.get("entities") or {}
                sentiment = entities.get("sentiment") if isinstance(entities, dict) else None
                if not isinstance(sentiment, dict):
                    continue
                basic = str(sentiment.get("basic") or "").strip().lower()
                if basic not in ("bullish", "bearish"):
                    continue

                ticker_count += 1
                if basic == "bullish":
                    ticker_bullish += 1
                else:
                    ticker_bearish += 1

                # Collect top posts for display (limit to 6 total)
                if len(posts) < 6:
                    body = str(msg.get("body") or "").strip()
                    created = str(msg.get("created_at") or "").strip() or None
                    user = msg.get("user") or {}
                    username = str(user.get("username") or "").strip() if isinstance(user, dict) else None
                    posts.append({
                        "title": body[:120] if body else f"${ticker} — {basic}",
                        "subreddit": f"StockTwits/${ticker}",
                        "url": f"https://stocktwits.com/symbol/{ticker}",
                        "upvotes": int(msg.get("likes", {}).get("total", 0)) if isinstance(msg.get("likes"), dict) else 0,
                        "num_comments": 0,
                        "created_at": created,
                        "excerpt": body[:280] if body else None,
                        "raw_score": 1.0 if basic == "bullish" else -1.0,
                        "sentiment_score": 100 if basic == "bullish" else 0,
                        "label": "Greed" if basic == "bullish" else "Fear",
                        "bullish_terms": 1 if basic == "bullish" else 0,
                        "bearish_terms": 1 if basic == "bearish" else 0,
                        "engagement": 1.0,
                    })

            total_bullish += ticker_bullish
            total_bearish += ticker_bearish
            total_messages += ticker_count

            breakdown.append({
                "subreddit": f"StockTwits/${ticker}",
                "post_count": ticker_count,
                "avg_sentiment_score": (
                    round((ticker_bullish / (ticker_bullish + ticker_bearish)) * 100, 1)
                    if (ticker_bullish + ticker_bearish) > 0
                    else None
                ),
                "note": None,
            })

        except Exception as exc:
            LOG.warning("StockTwits fetch failed for %s: %s", ticker, exc)
            breakdown.append({
                "subreddit": f"StockTwits/${ticker}",
                "post_count": 0,
                "avg_sentiment_score": None,
                "note": str(exc),
            })

    if total_bullish + total_bearish == 0:
        return _empty_social_payload(
            provider="stocktwits",
            now_utc=now_utc,
            tickers=tickers,
            note="No sentiment-tagged StockTwits messages found for tracked tickers",
            source_breakdown=breakdown,
        )

    raw_score = (total_bullish - total_bearish) / (total_bullish + total_bearish)
    score = _score_to_100(raw_score)

    return {
        "provider": "stocktwits",
        "source_type": "social",
        "available": True,
        "score": score,
        "raw_score": round(raw_score, 4),
        "label": _label_for_score(score),
        "post_count": total_messages,
        "subreddit_count": len(tickers),
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": 24,
        "subreddits": [],
        "tickers": tickers,
        "note": (
            f"{total_messages} sentiment-tagged StockTwits messages across {len(tickers)} tickers "
            f"({total_bullish} bullish, {total_bearish} bearish)"
        ),
        "bullish_terms_total": total_bullish,
        "bearish_terms_total": total_bearish,
        "posts": posts[:6],
        "source_breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_social_sentiment(*, now_utc: dt.datetime | None = None, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached social sentiment metadata.

    Uses StockTwits public API (no auth required, 200 req/hr).
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

    fresh_payload = _fetch_stocktwits_sentiment(resolved_now)
    expires_at = resolved_now + dt.timedelta(seconds=_cache_ttl_sec())

    with _cache_lock:
        _cache_payload = copy.deepcopy(fresh_payload)
        _cache_expires_at = expires_at

    return copy.deepcopy(fresh_payload)
