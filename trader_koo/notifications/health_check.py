"""Site health checker — pings all kooexperience.com services and alerts on failure."""

from __future__ import annotations

import logging
import time
from html import escape
from typing import Any

import httpx

from trader_koo.notifications.telegram import send_message

LOG = logging.getLogger(__name__)

SITES: list[dict[str, str]] = [
    {"name": "Portfolio", "url": "https://kooexperience.com"},
    {"name": "Trader Koo", "url": "https://trader.kooexperience.com/api/health"},
    {"name": "Job Hunter", "url": "https://job.kooexperience.com"},
    {"name": "LionWeather", "url": "https://lionweather.kooexperience.com"},
    {"name": "Photo ID Studio", "url": "https://studio.kooexperience.com"},
    {"name": "Wine Intelligence", "url": "https://wine.kooexperience.com"},
]

TIMEOUT_SEC = 15

# Track consecutive failures to avoid alert spam
_failure_counts: dict[str, int] = {}
_last_alert_at: dict[str, float] = {}
# Only alert after 2 consecutive failures (avoids transient blips)
ALERT_AFTER_FAILURES = 2
# Repeat a continuing outage no more than once every three hours.
HEALTH_ALERT_COOLDOWN_SEC = 3 * 3600


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
                _last_alert_at.pop(name, None)
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


def run_health_check(*, now_monotonic: float | None = None) -> None:
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
    now = time.monotonic() if now_monotonic is None else now_monotonic
    alertable = []
    for f in failures:
        count = _failure_counts.get(f["name"], 0)
        last_alert = _last_alert_at.get(f["name"])
        if count >= ALERT_AFTER_FAILURES and (
            last_alert is None or now - last_alert >= HEALTH_ALERT_COOLDOWN_SEC
        ):
            alertable.append(f)

    if not alertable:
        LOG.info(
            "Health check: %d failures but below alert threshold (%s)",
            len(failures),
            ", ".join(f"{f['name']}={_failure_counts.get(f['name'], 0)}" for f in failures),
        )
        return

    if _send_health_alert(alertable, results):
        for failure in alertable:
            _last_alert_at[failure["name"]] = now


def _send_health_alert(failures: list[dict], all_results: list[dict]) -> bool:
    """Send Telegram alert for site failures."""
    lines = ["<b>HEALTH CHECK FAILED</b>", ""]

    for f in failures:
        status = f.get("status", 0)
        error = f.get("error", "")
        count = _failure_counts.get(f["name"], 0)
        name = escape(str(f.get("name", "?")), quote=False)
        if status > 0:
            lines.append(f"  {name}: HTTP {status} ({count} consecutive)")
        else:
            lines.append(
                f"  {name}: {escape(str(error), quote=False)} ({count} consecutive)"
            )

    ok_count = sum(1 for r in all_results if r["ok"])
    lines.append("")
    lines.append(f"{ok_count}/{len(all_results)} sites healthy")

    text = "\n".join(lines)

    sent = send_message(text, parse_mode="HTML")
    if sent:
        LOG.info("Health alert sent for %d sites", len(failures))
    else:
        LOG.warning("Health alert Telegram send failed")
    return bool(sent)
