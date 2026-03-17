"""Optional Reddit-first social sentiment for the market dashboard.

This intentionally mirrors the old workflow-harvester archive scraper style:
curated subreddit targets, lightweight keyword gating, post-detail fetches,
and top-comment enrichment.
"""
from __future__ import annotations

import copy
import datetime as dt
import logging
import math
import os
import re
import threading
from typing import Any

import requests

LOG = logging.getLogger(__name__)

_DEFAULT_SUBREDDITS = (
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "SecurityAnalysis",
    "BitcoinMarkets",
    "CryptoCurrency",
    "ethtrader",
    "economy",
    "finance",
)
_DEFAULT_LOOKBACK_HOURS = 24
_DEFAULT_POST_LIMIT = 6
_DEFAULT_MIN_SCORE = 25
_DEFAULT_CACHE_TTL_SEC = 900
_TOP_COMMENT_LIMIT = 4

_MARKET_KEYWORDS = (
    "market",
    "stocks",
    "bullish",
    "bearish",
    "buy",
    "sell",
    "calls",
    "puts",
    "long",
    "short",
    "trade",
    "position",
    "breakout",
    "breakdown",
    "support",
    "resistance",
    "earnings",
    "fed",
    "inflation",
    "macro",
    "risk on",
    "risk off",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "crypto",
    "spy",
    "qqq",
    "volatility",
    "liquidity",
    "recession",
)

_BULLISH_TERMS = {
    "bull",
    "bullish",
    "buy",
    "calls",
    "long",
    "breakout",
    "rally",
    "rip",
    "moon",
    "squeeze",
    "accumulate",
    "upside",
    "strong",
    "beat",
    "support",
    "undervalued",
    "outperform",
}
_BEARISH_TERMS = {
    "bear",
    "bearish",
    "sell",
    "puts",
    "short",
    "dump",
    "crash",
    "rug",
    "breakdown",
    "recession",
    "overvalued",
    "miss",
    "weak",
    "resistance",
    "downside",
    "underperform",
}

_cache_lock = threading.Lock()
_cache_expires_at: dt.datetime | None = None
_cache_payload: dict[str, Any] | None = None


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
    return _int_env("TRADER_KOO_REDDIT_TIMEOUT_SEC", 15, lo=5, hi=60)


def _cache_ttl_sec() -> int:
    return _int_env("TRADER_KOO_SOCIAL_SENTIMENT_CACHE_TTL_SEC", _DEFAULT_CACHE_TTL_SEC, lo=60, hi=3600)


def _post_limit() -> int:
    return _int_env("TRADER_KOO_REDDIT_POST_LIMIT", _DEFAULT_POST_LIMIT, lo=3, hi=20)


def _min_score() -> int:
    return _int_env("TRADER_KOO_REDDIT_MIN_SCORE", _DEFAULT_MIN_SCORE, lo=1, hi=500)


def _user_agent() -> str:
    return str(
        os.getenv(
            "TRADER_KOO_REDDIT_USER_AGENT",
            "trader-koo/1.0 social sentiment (+https://trader.kooexperience.com)",
        )
    ).strip()


def _lookback_hours() -> int:
    return _DEFAULT_LOOKBACK_HOURS


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
    now_utc: dt.datetime,
    subreddits: list[str],
    note: str,
) -> dict[str, Any]:
    return {
        "provider": "reddit_public_json",
        "source_type": "social",
        "available": False,
        "score": None,
        "raw_score": None,
        "label": None,
        "post_count": 0,
        "subreddit_count": len(subreddits),
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": _lookback_hours(),
        "subreddits": subreddits,
        "note": note,
        "bullish_terms_total": 0,
        "bearish_terms_total": 0,
        "posts": [],
        "source_breakdown": [],
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())


def _matches_keywords(title: str, body: str) -> bool:
    text = f"{title} {body}".lower()
    return any(keyword in text for keyword in _MARKET_KEYWORDS)


def _lexicon_score(text: str) -> tuple[float | None, int, int]:
    tokens = _tokenize(text)
    if not tokens:
        return None, 0, 0
    bullish = sum(1 for token in tokens if token in _BULLISH_TERMS)
    bearish = sum(1 for token in tokens if token in _BEARISH_TERMS)
    if bullish == 0 and bearish == 0:
        return 0.0, 0, 0
    raw_score = (bullish - bearish) / max(bullish + bearish, 1)
    return raw_score, bullish, bearish


def _post_detail_url(permalink: str) -> str:
    if permalink.endswith("/"):
        return f"https://www.reddit.com{permalink}.json"
    return f"https://www.reddit.com{permalink}/.json"


def _fetch_post_detail(permalink: str) -> tuple[str, list[str]]:
    if not permalink:
        return "", []
    resp = requests.get(
        _post_detail_url(permalink),
        params={"raw_json": 1, "limit": _TOP_COMMENT_LIMIT},
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
        timeout=_request_timeout_sec(),
    )
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return "", []

    post_body = ""
    try:
        post_listing = ((payload[0] or {}).get("data") or {}).get("children") or []
        if post_listing and isinstance(post_listing[0], dict):
            post_data = post_listing[0].get("data") or {}
            post_body = str(post_data.get("selftext") or "").strip()
    except Exception:
        post_body = ""

    top_comments: list[str] = []
    try:
        comment_listing = ((payload[1] or {}).get("data") or {}).get("children") or []
        for child in comment_listing[:_TOP_COMMENT_LIMIT]:
            data = child.get("data") if isinstance(child, dict) else None
            if not isinstance(data, dict):
                continue
            body = str(data.get("body") or "").strip()
            score = max(int(data.get("score") or 0), 0)
            if body and score >= 5:
                top_comments.append(body)
    except Exception:
        top_comments = []

    return post_body, top_comments


def _normalize_post(post: dict[str, Any], subreddit: str) -> dict[str, Any] | None:
    title = str(post.get("title") or "").strip()
    body = str(post.get("selftext") or "").strip()
    if not title and not body:
        return None

    upvotes = max(int(post.get("score") or 0), 0)
    num_comments = max(int(post.get("num_comments") or 0), 0)
    if upvotes < _min_score():
        return None

    permalink = str(post.get("permalink") or "").strip()
    url = f"https://www.reddit.com{permalink}" if permalink else None
    detail_body, top_comments = _fetch_post_detail(permalink)
    body_parts = [body]
    if detail_body:
        body_parts.append(detail_body)
    if top_comments:
        body_parts.append("TOP COMMENTS:\n" + "\n---\n".join(top_comments))
    merged_body = "\n\n".join(part for part in body_parts if part).strip()

    if not _matches_keywords(title, merged_body):
        return None

    raw_score, bullish_terms, bearish_terms = _lexicon_score(f"{title}\n{merged_body}")
    engagement = 1.0 + math.log1p(upvotes) + 0.25 * math.log1p(num_comments)
    sentiment_score = _score_to_100(raw_score)
    created_utc = post.get("created_utc")
    created_at = None
    if isinstance(created_utc, int | float):
        created_at = _iso_utc(dt.datetime.fromtimestamp(created_utc, tz=dt.timezone.utc))

    return {
        "title": title,
        "subreddit": subreddit,
        "url": url,
        "upvotes": upvotes,
        "num_comments": num_comments,
        "created_at": created_at,
        "excerpt": merged_body[:280] if merged_body else None,
        "raw_score": round(raw_score or 0.0, 4),
        "sentiment_score": sentiment_score,
        "label": _label_for_score(sentiment_score),
        "bullish_terms": bullish_terms,
        "bearish_terms": bearish_terms,
        "engagement": round(engagement, 3),
    }


def _fetch_subreddit_posts(subreddit: str, *, limit: int) -> list[dict[str, Any]]:
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    resp = requests.get(
        url,
        params={"t": "day", "limit": limit, "raw_json": 1},
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
        timeout=_request_timeout_sec(),
    )
    resp.raise_for_status()
    payload = resp.json()
    listing = payload.get("data") or {}
    children = listing.get("children") or []
    posts: list[dict[str, Any]] = []
    for child in children:
        data = child.get("data") if isinstance(child, dict) else None
        if isinstance(data, dict):
            posts.append(data)
    return posts


def _fetch_reddit_social_sentiment(now_utc: dt.datetime) -> dict[str, Any]:
    subreddits = _csv_env("TRADER_KOO_REDDIT_SUBREDDITS", _DEFAULT_SUBREDDITS)
    post_limit = _post_limit()

    normalized_posts: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    weighted_total = 0.0
    weight_total = 0.0
    bullish_total = 0
    bearish_total = 0

    for subreddit in subreddits:
        try:
            posts = _fetch_subreddit_posts(subreddit, limit=post_limit)
        except Exception as exc:
            LOG.warning("Reddit social sentiment fetch failed for r/%s: %s", subreddit, exc)
            breakdown.append(
                {
                    "subreddit": subreddit,
                    "post_count": 0,
                    "avg_sentiment_score": None,
                    "note": str(exc),
                }
            )
            continue

        subreddit_scores: list[int] = []
        for post in posts:
            normalized = _normalize_post(post, subreddit)
            if not normalized:
                continue
            normalized_posts.append(normalized)
            bullish_total += int(normalized["bullish_terms"])
            bearish_total += int(normalized["bearish_terms"])
            subreddit_scores.append(int(normalized["sentiment_score"] or 50))
            weighted_total += float(normalized["raw_score"]) * float(normalized["engagement"])
            weight_total += float(normalized["engagement"])

        breakdown.append(
            {
                "subreddit": subreddit,
                "post_count": len(subreddit_scores),
                "avg_sentiment_score": round(sum(subreddit_scores) / len(subreddit_scores), 1)
                if subreddit_scores
                else None,
                "note": None,
            }
        )

    if weight_total <= 0 or not normalized_posts:
        return _empty_social_payload(
            now_utc=now_utc,
            subreddits=subreddits,
            note=(
                "No Reddit posts passed the archive-style subreddit, engagement, "
                "and keyword filters for the current window."
            ),
        )

    raw_score = weighted_total / weight_total
    score = _score_to_100(raw_score)
    normalized_posts.sort(
        key=lambda item: (float(item["engagement"]), int(item["upvotes"])),
        reverse=True,
    )

    return {
        "provider": "reddit_public_json",
        "source_type": "social",
        "available": True,
        "score": score,
        "raw_score": round(raw_score, 4),
        "label": _label_for_score(score),
        "post_count": len(normalized_posts),
        "subreddit_count": len(subreddits),
        "updated_at": _iso_utc(now_utc),
        "lookback_hours": _lookback_hours(),
        "subreddits": subreddits,
        "note": (
            f"{len(normalized_posts)} Reddit posts aggregated from curated market "
            "subreddits with post-detail and top-comment enrichment over the last 24h."
        ),
        "bullish_terms_total": bullish_total,
        "bearish_terms_total": bearish_total,
        "posts": normalized_posts[:6],
        "source_breakdown": breakdown,
    }


def get_social_sentiment(*, now_utc: dt.datetime | None = None, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached social sentiment metadata."""
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

    fresh_payload = _fetch_reddit_social_sentiment(resolved_now)
    expires_at = resolved_now + dt.timedelta(seconds=_cache_ttl_sec())

    with _cache_lock:
        _cache_payload = copy.deepcopy(fresh_payload)
        _cache_expires_at = expires_at

    return copy.deepcopy(fresh_payload)
