"""Prediction market + crypto spike detection and Telegram alerting.

Archives Polymarket probabilities hourly, detects sudden moves in both
prediction markets and crypto prices/OI, and sends formatted Telegram
alerts when spikes exceed configurable thresholds.

Public API
----------
``snapshot_polymarket(db_path) -> int``
``detect_polymarket_spikes(db_path, lookback_hours, threshold_pct) -> list[dict]``
``detect_crypto_spikes(db_path, lookback_hours) -> list[dict]``
``send_spike_alerts(db_path) -> int``
``ensure_polymarket_schema(conn) -> None``
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import sqlite3
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from trader_koo.config import env_float, env_int

LOG = logging.getLogger("trader_koo.notifications.market_monitor")

POLYMARKET_MIN_ALERT_VOLUME_USD = 25_000.0
POLYMARKET_MIN_ALERT_LIQUIDITY_USD = 25_000.0
POLYMARKET_MIN_ALERT_VOLUME_24H_USD = 5_000.0
POLYMARKET_MAX_SNAPSHOT_AGE_MINUTES = 15
POLYMARKET_BASELINE_TOLERANCE_MINUTES = 15
POLYMARKET_MAX_ALERT_GROUPS = env_int(
    "TRADER_KOO_POLYMARKET_MAX_ALERT_GROUPS", 4, min_value=1, max_value=10,
)
POLYMARKET_GROUP_COOLDOWN_HOURS = env_float(
    "TRADER_KOO_POLYMARKET_GROUP_COOLDOWN_HOURS", 12.0, min_value=1.0, max_value=168.0,
)
POLYMARKET_GROUP_BREAKTHROUGH_DELTA_PTS = env_float(
    "TRADER_KOO_POLYMARKET_GROUP_BREAKTHROUGH_DELTA_PTS", 10.0, min_value=1.0, max_value=50.0,
)

_POLYMARKET_ASSET_ALIASES = {
    "BTC": ("bitcoin", "btc"),
    "ETH": ("ethereum", "ether", "eth"),
    "SOL": ("solana", "sol"),
    "XRP": ("xrp",),
    "DOGE": ("dogecoin", "doge"),
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_polymarket_schema(conn: sqlite3.Connection) -> None:
    """Create the polymarket_snapshots table and index if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS polymarket_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_slug TEXT NOT NULL,
            event_title TEXT NOT NULL,
            market_question TEXT NOT NULL,
            probability REAL NOT NULL,
            volume REAL,
            snapshot_ts TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_poly_snap_slug_ts
        ON polymarket_snapshots(event_slug, snapshot_ts)
    """)
    conn.commit()
    LOG.info("polymarket_snapshots schema ensured")


# ---------------------------------------------------------------------------
# Snapshot archival
# ---------------------------------------------------------------------------

def snapshot_polymarket(db_path: Path) -> int:
    """Fetch current Polymarket events and archive probabilities.

    For each active sub-market within each event, stores the current
    YES probability and volume.  Returns the number of snapshots saved.
    """
    from trader_koo.ml.external_data import fetch_polymarket_events

    try:
        # Archival must observe the provider again. Re-stamping the one-hour
        # display cache every five minutes would fabricate snapshot freshness.
        events = fetch_polymarket_events(limit=50, use_cache=False)
    except Exception as exc:
        LOG.error("Failed to fetch Polymarket events for snapshot: %s", exc)
        return 0

    if not events:
        LOG.warning("No Polymarket events returned for snapshot")
        return 0

    now_iso = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    rows_to_insert: list[tuple[str, str, str, float, float | None, str]] = []

    for event in events:
        slug = event.get("slug", "")
        title = event.get("title", "")
        markets = event.get("markets") or []

        for mkt in markets:
            if not mkt.get("active", False):
                continue

            question = mkt.get("question", "")
            prices = mkt.get("prices_pct") or []
            outcomes = mkt.get("outcomes") or []
            volume = mkt.get("volume")
            liquidity = float(mkt.get("liquidity") or 0)
            volume_24h = float(mkt.get("volume_24h") or 0)
            if (
                liquidity < POLYMARKET_MIN_ALERT_LIQUIDITY_USD
                or volume_24h < POLYMARKET_MIN_ALERT_VOLUME_24H_USD
            ):
                continue

            # Extract YES probability
            yes_prob: float | None = None
            for outcome, price in zip(outcomes, prices):
                if str(outcome).lower() == "yes" and price is not None:
                    yes_prob = float(price)
                    break
            # Fallback: first price if no explicit YES
            if yes_prob is None and prices and prices[0] is not None:
                yes_prob = float(prices[0])

            if yes_prob is None:
                continue

            rows_to_insert.append((
                slug, title, question, yes_prob, volume, now_iso,
            ))

    if not rows_to_insert:
        LOG.info("No active markets to snapshot")
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_polymarket_schema(conn)
        conn.executemany(
            """
            INSERT INTO polymarket_snapshots
                (event_slug, event_title, market_question, probability, volume, snapshot_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
        conn.commit()
        LOG.info("Saved %d Polymarket snapshots at %s", len(rows_to_insert), now_iso)
        return len(rows_to_insert)
    except Exception as exc:
        LOG.error("Failed to save Polymarket snapshots: %s", exc)
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Polymarket spike detection
# ---------------------------------------------------------------------------

def detect_polymarket_spikes(
    db_path: Path,
    lookback_hours: int = 6,
    threshold_pct: float = 5.0,
    *,
    min_volume_usd: float = POLYMARKET_MIN_ALERT_VOLUME_USD,
    max_snapshot_age_minutes: int = POLYMARKET_MAX_SNAPSHOT_AGE_MINUTES,
    baseline_tolerance_minutes: int = POLYMARKET_BASELINE_TOLERANCE_MINUTES,
    now_utc: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    """Compare current probabilities to ``lookback_hours`` ago.

    Returns a list of spike dicts for any market where the absolute
    probability change exceeds ``threshold_pct`` percentage points.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ensure_polymarket_schema(conn)

        now = (now_utc or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        # Get latest snapshot per market
        latest_rows = conn.execute("""
            SELECT event_slug, event_title, market_question,
                   probability, volume, snapshot_ts
            FROM polymarket_snapshots
            WHERE snapshot_ts = (
                SELECT MAX(snapshot_ts) FROM polymarket_snapshots
            )
        """).fetchall()

        if not latest_rows:
            return []

        latest_ts_raw = str(latest_rows[0]["snapshot_ts"] or "")
        try:
            latest_ts = dt.datetime.fromisoformat(latest_ts_raw.replace("Z", "+00:00"))
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=dt.timezone.utc)
            latest_ts = latest_ts.astimezone(dt.timezone.utc)
        except ValueError:
            LOG.warning("Polymarket spike detection skipped: invalid latest snapshot timestamp")
            return []
        snapshot_age = (now - latest_ts).total_seconds()
        if snapshot_age < -300 or snapshot_age > max_snapshot_age_minutes * 60:
            LOG.warning(
                "Polymarket spike detection skipped: latest snapshot is stale (%s)",
                latest_ts_raw,
            )
            return []

        cutoff = latest_ts - dt.timedelta(hours=lookback_hours)
        baseline_floor = cutoff - dt.timedelta(minutes=baseline_tolerance_minutes)

        spikes: list[dict[str, Any]] = []
        for row in latest_rows:
            slug = row["event_slug"]
            question = row["market_question"]
            new_prob = row["probability"]
            volume = row["volume"]
            title = row["event_title"]

            if volume is None or float(volume) < min_volume_usd:
                continue

            # Find the closest snapshot at or before the lookback cutoff.
            old_row = conn.execute(
                """
                SELECT probability, snapshot_ts
                FROM polymarket_snapshots
                WHERE event_slug = ? AND market_question = ?
                      AND snapshot_ts BETWEEN ? AND ?
                ORDER BY snapshot_ts DESC
                LIMIT 1
                """,
                (slug, question, baseline_floor.isoformat(), cutoff.isoformat()),
            ).fetchone()

            if old_row is None:
                continue

            old_prob = old_row["probability"]
            change = new_prob - old_prob

            if abs(change) >= threshold_pct:
                direction = "up" if change > 0 else "down"
                spikes.append({
                    "event_title": title,
                    "event_slug": slug,
                    "question": question,
                    "old_prob": round(old_prob, 1),
                    "new_prob": round(new_prob, 1),
                    "change_pct": round(change, 1),
                    "direction": direction,
                    "volume": volume,
                    "lookback_hours": lookback_hours,
                    "baseline_snapshot_ts": old_row["snapshot_ts"],
                    "latest_snapshot_ts": latest_ts_raw,
                })

        spikes.sort(key=lambda s: abs(s["change_pct"]), reverse=True)
        LOG.info(
            "Polymarket spike detection: %d spikes found (lookback=%dh, threshold=%.1f%%)",
            len(spikes), lookback_hours, threshold_pct,
        )
        return spikes
    except Exception as exc:
        LOG.error("Polymarket spike detection failed: %s", exc)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Crypto spike detection
# ---------------------------------------------------------------------------

_CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
_CRYPTO_PRICE_THRESHOLD_PCT = 5.0
_CRYPTO_OI_THRESHOLD_PCT = 10.0


def detect_crypto_spikes(
    db_path: Path,
    lookback_hours: int = 4,
) -> list[dict[str, Any]]:
    """Detect crypto price and open-interest spikes.

    Checks ``crypto_bars`` for price moves exceeding 5% and queries
    Binance OI history for changes exceeding 10%.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    spikes: list[dict[str, Any]] = []

    try:
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = (now - dt.timedelta(hours=lookback_hours)).isoformat()

        # Check if crypto_bars table exists
        table_check = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='crypto_bars' LIMIT 1"
        ).fetchone()
        if not table_check:
            LOG.warning("crypto_bars table not found — skipping crypto spike detection")
            return []

        for symbol in _CRYPTO_SYMBOLS:
            # Get latest bar
            latest = conn.execute(
                """
                SELECT close, timestamp FROM crypto_bars
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()

            if not latest:
                continue

            current_price = float(latest["close"])

            # Get bar from lookback window
            old_bar = conn.execute(
                """
                SELECT close, timestamp FROM crypto_bars
                WHERE symbol = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol, cutoff),
            ).fetchone()

            if not old_bar:
                continue

            old_price = float(old_bar["close"])
            if old_price <= 0:
                continue

            price_change_pct = ((current_price - old_price) / old_price) * 100

            # Check OI via Binance API
            oi_change_pct: float | None = None
            try:
                from trader_koo.crypto.binance_oi import fetch_open_interest_history

                oi_snapshots = fetch_open_interest_history(
                    symbol, period="1h", limit=max(lookback_hours + 2, 10),
                )
                if len(oi_snapshots) >= 2:
                    latest_oi = oi_snapshots[-1].sum_open_interest_value
                    # Find OI from approximately lookback_hours ago
                    target_ts = now - dt.timedelta(hours=lookback_hours)
                    old_oi_snap = min(
                        oi_snapshots,
                        key=lambda s: abs((s.timestamp - target_ts).total_seconds()),
                    )
                    old_oi = old_oi_snap.sum_open_interest_value
                    if old_oi > 0:
                        oi_change_pct = ((latest_oi - old_oi) / old_oi) * 100
            except Exception as exc:
                LOG.debug("OI fetch failed for %s: %s", symbol, exc)

            # Determine if either threshold is breached
            price_spike = abs(price_change_pct) >= _CRYPTO_PRICE_THRESHOLD_PCT
            oi_spike = oi_change_pct is not None and abs(oi_change_pct) >= _CRYPTO_OI_THRESHOLD_PCT

            if price_spike or oi_spike:
                direction = "up" if price_change_pct > 0 else "down"
                spike: dict[str, Any] = {
                    "symbol": symbol,
                    "old_price": round(old_price, 2),
                    "new_price": round(current_price, 2),
                    "price_change_pct": round(price_change_pct, 1),
                    "direction": direction,
                    "lookback_hours": lookback_hours,
                    "price_spike": price_spike,
                    "oi_spike": oi_spike,
                }
                if oi_change_pct is not None:
                    spike["oi_change_pct"] = round(oi_change_pct, 1)
                spikes.append(spike)

        spikes.sort(key=lambda s: abs(s["price_change_pct"]), reverse=True)
        LOG.info(
            "Crypto spike detection: %d spikes found (lookback=%dh)",
            len(spikes), lookback_hours,
        )
        return spikes
    except Exception as exc:
        LOG.error("Crypto spike detection failed: %s", exc)
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Telegram alert formatting + sending
# ---------------------------------------------------------------------------

def _format_volume(vol: float | None) -> str:
    """Format volume as human-readable string."""
    if vol is None or vol <= 0:
        return "N/A"
    if vol >= 1_000_000:
        return f"${vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"${vol / 1_000:.1f}K"
    return f"${vol:.0f}"


def _html(value: Any) -> str:
    """Escape external text before inserting it into Telegram HTML."""
    return escape(str(value), quote=False)


def _polymarket_group_key(spike: dict[str, Any]) -> str:
    """Return a stable topic key so correlated contracts share one alert slot."""
    text = " ".join(
        str(spike.get(field) or "")
        for field in ("event_title", "question", "event_slug")
    ).lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    for asset, aliases in _POLYMARKET_ASSET_ALIASES.items():
        if any(alias in words for alias in aliases):
            return f"asset:{asset}"

    slug = str(spike.get("event_slug") or "").strip().lower()
    if slug:
        return f"event:{slug}"
    title = re.sub(r"[^a-z0-9]+", "-", str(spike.get("event_title") or "market").lower())
    return f"title:{title.strip('-') or 'market'}"


def _select_polymarket_digest(
    spikes: list[dict[str, Any]],
    *,
    max_groups: int = POLYMARKET_MAX_ALERT_GROUPS,
) -> list[dict[str, Any]]:
    """Choose one strongest, most liquid representative per market topic."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spike in spikes:
        grouped.setdefault(_polymarket_group_key(spike), []).append(spike)

    selected: list[dict[str, Any]] = []
    for group_key, members in grouped.items():
        members.sort(
            key=lambda item: (
                abs(float(item.get("change_pct") or 0)),
                float(item.get("volume") or 0),
            ),
            reverse=True,
        )
        representative = dict(members[0])
        representative["alert_group_key"] = group_key
        representative["related_count"] = len(members) - 1
        selected.append(representative)

    selected.sort(
        key=lambda item: (
            abs(float(item.get("change_pct") or 0)),
            float(item.get("volume") or 0),
        ),
        reverse=True,
    )
    return selected[:max_groups]


def send_spike_alerts(db_path: Path) -> int:
    """Run both spike detectors and send Telegram alerts.

    Returns the total number of alerts sent.
    """
    from trader_koo.notifications.telegram import is_configured, send_message

    if not is_configured():
        LOG.info("Telegram not configured — skipping spike alerts")
        return 0

    alerts_sent = 0

    # Collect all spikes, filter out already-alerted ones, send ONE message
    all_lines: list[str] = []

    # Cooldown: track which events we already alerted (by slug+direction)
    # Only re-alert if direction CHANGES or probability moves another 5+ pts
    conn_cd = sqlite3.connect(str(db_path))
    try:
        conn_cd.execute("""
            CREATE TABLE IF NOT EXISTS spike_alert_cooldown (
                event_key TEXT PRIMARY KEY,
                direction TEXT,
                last_prob REAL,
                alerted_at TEXT
            )
        """)
        conn_cd.commit()
    except Exception:
        pass

    def _should_alert(
        key: str,
        direction: str,
        new_prob: float,
        *,
        legacy_key: str | None = None,
    ) -> bool:
        """Only alert if direction changed or prob moved 5+ pts since last alert."""
        keys = (key, legacy_key) if legacy_key and legacy_key != key else (key, key)
        row = conn_cd.execute(
            """
            SELECT direction, last_prob
            FROM spike_alert_cooldown
            WHERE event_key IN (?, ?)
            ORDER BY CASE WHEN event_key = ? THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (*keys, key),
        ).fetchone()
        if row is None:
            return True  # never alerted
        old_dir, old_prob = row
        if old_dir != direction:
            return True  # direction reversed
        if abs(new_prob - (old_prob or 0)) >= 5.0:
            return True  # moved another 5+ pts
        return False  # same direction, small move — skip

    def _mark_alerted(
        key: str,
        direction: str,
        prob: float,
        *,
        legacy_key: str | None = None,
    ) -> None:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        conn_cd.execute(
            "INSERT OR REPLACE INTO spike_alert_cooldown VALUES (?, ?, ?, ?)",
            (key, direction, prob, now_iso),
        )
        if legacy_key and legacy_key != key:
            conn_cd.execute(
                "DELETE FROM spike_alert_cooldown WHERE event_key = ?",
                (legacy_key,),
            )

    pending_cooldowns: list[tuple[str, str, float, str | None]] = []

    # Polymarket spikes
    try:
        poly_spikes = _select_polymarket_digest(detect_polymarket_spikes(db_path))
        for spike in poly_spikes:
            slug = str(spike.get("event_slug") or "")
            question = str(spike.get("question") or spike.get("event_title") or "?")
            direction = spike.get("direction", "up")
            new_p = spike.get("new_prob", 0)
            group_key = str(spike["alert_group_key"])
            key = f"polymarket:{group_key}"
            magnitude = abs(float(spike.get("change_pct") or 0))
            row = conn_cd.execute(
                "SELECT last_prob, alerted_at FROM spike_alert_cooldown WHERE event_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                try:
                    alerted_at = dt.datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
                    if alerted_at.tzinfo is None:
                        alerted_at = alerted_at.replace(tzinfo=dt.timezone.utc)
                    elapsed = dt.datetime.now(dt.timezone.utc) - alerted_at.astimezone(dt.timezone.utc)
                except (TypeError, ValueError):
                    elapsed = dt.timedelta.max
                previous_magnitude = float(row[0] or 0)
                still_cooling = elapsed < dt.timedelta(hours=POLYMARKET_GROUP_COOLDOWN_HOURS)
                is_breakthrough = magnitude >= previous_magnitude + POLYMARKET_GROUP_BREAKTHROUGH_DELTA_PTS
                if still_cooling and not is_breakthrough:
                    continue

            arrow = "\u2B06\uFE0F" if direction == "up" else "\u2B07\uFE0F"
            old_p = spike.get("old_prob", 0)
            change = spike.get("change_pct", 0)
            vol = _format_volume(spike.get("volume", 0))
            poly_link = f"https://polymarket.com/event/{quote(slug, safe='')}" if slug else ""
            link_html = f'\n   <a href="{poly_link}">View on Polymarket</a>' if slug else ""
            related_count = int(spike.get("related_count") or 0)
            related_html = f" | +{related_count} related" if related_count else ""
            all_lines.append(
                f"{arrow} <b>{_html(question)}</b>\n"
                f"   {old_p:.0f}% \u2192 {new_p:.0f}% ({change:+.1f} pts) | lifetime vol {vol}"
                f"{related_html}"
                f"{link_html}"
            )
            pending_cooldowns.append((key, "digest", magnitude, None))
    except Exception as exc:
        LOG.error("Polymarket spike alerting failed: %s", exc)

    # Crypto spikes
    try:
        crypto_spikes = detect_crypto_spikes(db_path)
        for spike in crypto_spikes:
            sym = str(spike.get("symbol") or "?")
            direction = spike.get("direction", "up" if spike.get("price_change_pct", 0) > 0 else "down")
            new_price = spike.get("new_price", 0)
            key = f"crypto:{sym}"

            if not _should_alert(key, direction, new_price):
                continue

            arrow = "\U0001F4C8" if direction == "up" else "\U0001F4C9"
            price_chg = spike.get("price_change_pct", 0)
            oi_chg = spike.get("oi_change_pct")
            parts = [f"{arrow} {_html(sym)}: {price_chg:+.1f}%"]
            if oi_chg is not None and spike.get("oi_spike"):
                parts.append(f"OI {oi_chg:+.0f}%")
            all_lines.append(" | ".join(parts))
            pending_cooldowns.append((key, direction, new_price, None))
    except Exception as exc:
        LOG.error("Crypto spike alerting failed: %s", exc)

    # Send ONE compiled message (HTML for clickable links)
    if all_lines:
        event_label = "event" if len(all_lines) == 1 else "events"
        header = f"\U0001F6A8 <b>Market Spikes ({len(all_lines)} {event_label})</b>\n"
        body = "\n\n".join(all_lines)
        footer = '\n\n<a href="https://trader.kooexperience.com/markets">View all on Dashboard</a>'
        msg = f"{header}\n{body}{footer}"
        if send_message(msg, parse_mode="HTML"):
            for key, direction, value, legacy_key in pending_cooldowns:
                _mark_alerted(
                    key,
                    direction,
                    value,
                    legacy_key=legacy_key,
                )
            conn_cd.commit()
            alerts_sent = len(all_lines)
        else:
            LOG.warning("Failed to send compiled spike alert")

    conn_cd.close()
    LOG.info("Spike alerts: %d events in %d messages", alerts_sent, 1 if all_lines else 0)
    return alerts_sent


# ---------------------------------------------------------------------------
# Recent spikes query (for admin API)
# ---------------------------------------------------------------------------

def get_recent_spikes(
    db_path: Path,
    hours: int = 24,
) -> dict[str, Any]:
    """Return spike detection results from the last N hours.

    Runs both detectors live and returns their combined output.
    """
    poly_spikes: list[dict[str, Any]] = []
    crypto_spikes: list[dict[str, Any]] = []

    try:
        poly_spikes = detect_polymarket_spikes(db_path, lookback_hours=min(hours, 12))
    except Exception as exc:
        LOG.error("Failed to get recent Polymarket spikes: %s", exc)

    try:
        crypto_spikes = detect_crypto_spikes(db_path, lookback_hours=min(hours, 12))
    except Exception as exc:
        LOG.error("Failed to get recent crypto spikes: %s", exc)

    # Get snapshot stats
    snapshot_count = 0
    latest_snapshot_ts: str | None = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cutoff = (
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
            ).isoformat()
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt, MAX(snapshot_ts) AS latest_ts
                FROM polymarket_snapshots
                WHERE snapshot_ts >= ?
                """,
                (cutoff,),
            ).fetchone()
            if row:
                snapshot_count = int(row["cnt"] or 0)
                latest_snapshot_ts = row["latest_ts"]
        finally:
            conn.close()
    except Exception:
        pass

    return {
        "ok": True,
        "lookback_hours": hours,
        "polymarket_spikes": poly_spikes,
        "polymarket_spike_count": len(poly_spikes),
        "crypto_spikes": crypto_spikes,
        "crypto_spike_count": len(crypto_spikes),
        "total_spikes": len(poly_spikes) + len(crypto_spikes),
        "snapshots_in_window": snapshot_count,
        "latest_snapshot_ts": latest_snapshot_ts,
        "checked_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
