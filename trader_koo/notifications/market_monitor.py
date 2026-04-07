"""Prediction market + crypto spike detection and Telegram alerting.

Archives Polymarket probabilities hourly, detects sudden moves in both
prediction markets and crypto prices/OI, and sends formatted Telegram
alerts when spikes exceed configurable thresholds.

Public API
----------
``snapshot_polymarket(db_path) -> int``
``detect_polymarket_spikes(db_path, lookback_hours, threshold_pct) -> list[dict]``
``detect_crypto_spikes(db_path, lookback_hours) -> list[dict]``
``send_spike_alerts(db_path, report_dir) -> int``
``ensure_polymarket_schema(conn) -> None``
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from pathlib import Path
from typing import Any

LOG = logging.getLogger("trader_koo.notifications.market_monitor")


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
        events = fetch_polymarket_events(limit=50)
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
) -> list[dict[str, Any]]:
    """Compare current probabilities to ``lookback_hours`` ago.

    Returns a list of spike dicts for any market where the absolute
    probability change exceeds ``threshold_pct`` percentage points.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ensure_polymarket_schema(conn)

        now = dt.datetime.now(dt.timezone.utc)
        cutoff = (now - dt.timedelta(hours=lookback_hours)).isoformat()

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

        spikes: list[dict[str, Any]] = []
        for row in latest_rows:
            slug = row["event_slug"]
            question = row["market_question"]
            new_prob = row["probability"]
            volume = row["volume"]
            title = row["event_title"]

            # Find the oldest snapshot within lookback window for this market
            old_row = conn.execute(
                """
                SELECT probability, snapshot_ts
                FROM polymarket_snapshots
                WHERE event_slug = ? AND market_question = ?
                      AND snapshot_ts <= ?
                ORDER BY snapshot_ts ASC
                LIMIT 1
                """,
                (slug, question, cutoff),
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


def _format_price(price: float) -> str:
    """Format price with commas."""
    if price >= 1:
        return f"${price:,.2f}"
    return f"${price:.4f}"


def _format_polymarket_alert(spike: dict[str, Any]) -> str:
    """Format a single Polymarket spike as a Telegram message."""
    arrow = "\u2b06\ufe0f" if spike["direction"] == "up" else "\u2b07\ufe0f"
    sign = "+" if spike["change_pct"] > 0 else ""
    vol_str = _format_volume(spike.get("volume"))

    return (
        "\U0001f6a8 *Prediction Market Spike*\n"
        "\n"
        f"\U0001f4ca {spike['event_title']}\n"
        f"{arrow} {spike['old_prob']:.0f}% \u2192 {spike['new_prob']:.0f}% "
        f"({sign}{spike['change_pct']:.1f} pts in {spike['lookback_hours']}h)\n"
        f"\U0001f4b0 Vol: {vol_str}\n"
        "\n"
        "Possible signal: insider activity or breaking news"
    )


def _format_crypto_alert(spike: dict[str, Any]) -> str:
    """Format a single crypto spike as a Telegram message."""
    arrow = "\U0001f4c8" if spike["direction"] == "up" else "\U0001f4c9"
    sign = "+" if spike["price_change_pct"] > 0 else ""

    lines = [
        "\U0001f6a8 *Crypto Alert*",
        "",
        f"{arrow} {spike['symbol']}: {_format_price(spike['old_price'])} \u2192 "
        f"{_format_price(spike['new_price'])} "
        f"({sign}{spike['price_change_pct']:.1f}% in {spike['lookback_hours']}h)",
    ]

    if spike.get("oi_change_pct") is not None:
        oi_sign = "+" if spike["oi_change_pct"] > 0 else ""
        lines.append(
            f"\U0001f4ca OI change: {oi_sign}{spike['oi_change_pct']:.0f}%"
        )

    if spike.get("price_spike") and spike.get("oi_spike"):
        lines.append("Volume surge detected")

    return "\n".join(lines)


def send_spike_alerts(db_path: Path, report_dir: Path) -> int:
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

    def _should_alert(key: str, direction: str, new_prob: float) -> bool:
        """Only alert if direction changed or prob moved 5+ pts since last alert."""
        row = conn_cd.execute(
            "SELECT direction, last_prob FROM spike_alert_cooldown WHERE event_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return True  # never alerted
        old_dir, old_prob = row
        if old_dir != direction:
            return True  # direction reversed
        if abs(new_prob - (old_prob or 0)) >= 5.0:
            return True  # moved another 5+ pts
        return False  # same direction, small move — skip

    def _mark_alerted(key: str, direction: str, prob: float) -> None:
        now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        conn_cd.execute(
            "INSERT OR REPLACE INTO spike_alert_cooldown VALUES (?, ?, ?, ?)",
            (key, direction, prob, now_iso),
        )
        conn_cd.commit()

    # Polymarket spikes
    try:
        poly_spikes = detect_polymarket_spikes(db_path)
        for spike in poly_spikes:
            slug = spike.get("event_slug", "")
            question = spike.get("question", spike.get("event_title", "?"))
            direction = spike.get("direction", "up")
            new_p = spike.get("new_prob", 0)
            key = f"{slug}:{question[:60]}"

            if not _should_alert(key, direction, new_p):
                continue  # already alerted, same direction, small move

            arrow = "\u2B06\uFE0F" if direction == "up" else "\u2B07\uFE0F"
            old_p = spike.get("old_prob", 0)
            change = spike.get("change_pct", 0)
            vol = _format_volume(spike.get("volume", 0))
            poly_link = f"https://polymarket.com/event/{slug}" if slug else ""
            link_html = f'\n   <a href="{poly_link}">View on Polymarket</a>' if slug else ""
            all_lines.append(
                f"{arrow} <b>{question}</b>\n"
                f"   {old_p:.0f}% \u2192 {new_p:.0f}% ({change:+.1f} pts) | {vol}"
                f"{link_html}"
            )
            _mark_alerted(key, direction, new_p)
    except Exception as exc:
        LOG.error("Polymarket spike alerting failed: %s", exc)

    # Crypto spikes
    try:
        crypto_spikes = detect_crypto_spikes(db_path)
        for spike in crypto_spikes:
            sym = spike.get("symbol", "?")
            direction = spike.get("direction", "up" if spike.get("price_change_pct", 0) > 0 else "down")
            new_price = spike.get("new_price", 0)
            key = f"crypto:{sym}"

            if not _should_alert(key, direction, new_price):
                continue

            arrow = "\U0001F4C8" if direction == "up" else "\U0001F4C9"
            price_chg = spike.get("price_change_pct", 0)
            oi_chg = spike.get("oi_change_pct")
            parts = [f"{arrow} {sym}: {price_chg:+.1f}%"]
            if oi_chg is not None and spike.get("oi_spike"):
                parts.append(f"OI {oi_chg:+.0f}%")
            all_lines.append(" | ".join(parts))
            _mark_alerted(key, direction, new_price)
    except Exception as exc:
        LOG.error("Crypto spike alerting failed: %s", exc)

    conn_cd.close()

    # Send ONE compiled message (HTML for clickable links)
    if all_lines:
        header = f"\U0001F6A8 <b>Market Spikes ({len(all_lines)} events)</b>\n"
        body = "\n\n".join(all_lines)
        footer = '\n\n<a href="https://trader.kooexperience.com/markets">View all on Dashboard</a>'
        msg = f"{header}\n{body}{footer}"
        if send_message(msg, parse_mode="HTML"):
            alerts_sent = len(all_lines)
        else:
            LOG.warning("Failed to send compiled spike alert")

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
