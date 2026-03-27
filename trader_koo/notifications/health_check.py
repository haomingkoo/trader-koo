"""Site health checker — pings all kooexperience.com services and alerts on failure."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

LOG = logging.getLogger(__name__)

SITES: list[dict[str, str]] = [
    {"name": "Portfolio", "url": "https://kooexperience.com"},
    {"name": "Trader Koo", "url": "https://trader.kooexperience.com/api/status"},
    {"name": "Job Hunter", "url": "https://job.kooexperience.com"},
    {"name": "LionWeather", "url": "https://lionweather.kooexperience.com"},
    {"name": "Photo ID Studio", "url": "https://studio.kooexperience.com"},
    {"name": "Wine Intelligence", "url": "https://wine.kooexperience.com"},
]

TIMEOUT_SEC = 15

# Track consecutive failures to avoid alert spam
_failure_counts: dict[str, int] = {}
# Only alert after 2 consecutive failures (avoids transient blips)
ALERT_AFTER_FAILURES = 2
# Alert again every N failures after the first alert
REPEAT_ALERT_EVERY = 6  # ~3 hours at 30min intervals


def check_all_sites() -> list[dict[str, Any]]:
    """Ping all sites. Returns list of results."""
    results: list[dict[str, Any]] = []
    for site in SITES:
        name = site["name"]
        url = site["url"]
        start = time.monotonic()
        try:
            resp = httpx.get(url, timeout=TIMEOUT_SEC, follow_redirects=True)
            elapsed_ms = (time.monotonic() - start) * 1000
            ok = 200 <= resp.status_code < 400
            results.append({
                "name": name,
                "url": url,
                "status": resp.status_code,
                "ok": ok,
                "elapsed_ms": round(elapsed_ms),
            })
            if ok:
                _failure_counts[name] = 0
            else:
                _failure_counts[name] = _failure_counts.get(name, 0) + 1
                LOG.warning("Health check FAIL: %s returned %d", name, resp.status_code)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            _failure_counts[name] = _failure_counts.get(name, 0) + 1
            results.append({
                "name": name,
                "url": url,
                "status": 0,
                "ok": False,
                "elapsed_ms": round(elapsed_ms),
                "error": str(exc)[:100],
            })
            LOG.warning("Health check FAIL: %s — %s", name, exc)

    return results


def run_health_check() -> None:
    """Run health check and send Telegram alert for any failures."""
    results = check_all_sites()
    failures = [r for r in results if not r["ok"]]

    if not failures:
        LOG.info(
            "Health check OK: all %d sites up (%s)",
            len(results),
            ", ".join(f"{r['name']} {r['elapsed_ms']}ms" for r in results),
        )
        return

    # Only alert after consecutive failures to avoid transient noise
    alertable = []
    for f in failures:
        count = _failure_counts.get(f["name"], 0)
        if count == ALERT_AFTER_FAILURES or (count > ALERT_AFTER_FAILURES and count % REPEAT_ALERT_EVERY == 0):
            alertable.append(f)

    if not alertable:
        LOG.info(
            "Health check: %d failures but below alert threshold (%s)",
            len(failures),
            ", ".join(f"{f['name']}={_failure_counts.get(f['name'], 0)}" for f in failures),
        )
        return

    _send_health_alert(alertable, results)


def _send_health_alert(failures: list[dict], all_results: list[dict]) -> None:
    """Send Telegram alert for site failures."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not bot_token or not chat_id:
        return

    lines = ["<b>SITE DOWN</b>", ""]

    for f in failures:
        status = f.get("status", 0)
        error = f.get("error", "")
        count = _failure_counts.get(f["name"], 0)
        if status > 0:
            lines.append(f"  {f['name']}: HTTP {status} ({count} consecutive)")
        else:
            lines.append(f"  {f['name']}: {error} ({count} consecutive)")

    ok_count = sum(1 for r in all_results if r["ok"])
    lines.append("")
    lines.append(f"{ok_count}/{len(all_results)} sites healthy")

    text = "\n".join(lines)

    try:
        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        LOG.info("Health alert sent for %d sites", len(failures))
    except Exception as exc:
        LOG.debug("Health alert Telegram failed: %s", exc)
