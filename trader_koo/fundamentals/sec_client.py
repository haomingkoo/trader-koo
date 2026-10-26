"""Rate-limited SEC EDGAR client (fair access: declared User-Agent, <=10 req/s)."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import requests

from trader_koo.fundamentals import config as cfg

LOG = logging.getLogger("trader_koo.fundamentals.sec")


class SecFetchError(RuntimeError):
    """Raised when SEC keeps refusing or failing after all retries."""


class SecClient:
    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        requests_per_second: float = cfg.REQUESTS_PER_SECOND,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": cfg.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
        self._sleep = sleep
        self._clock = clock
        self._interval = 1.0 / requests_per_second
        self._next_slot = 0.0

    def _throttle(self) -> None:
        now = self._clock()
        if now < self._next_slot:
            self._sleep(self._next_slot - now)
            now = self._next_slot
        self._next_slot = now + self._interval

    def get_json(self, url: str) -> tuple[int, Any | None]:
        """Return (status, payload). 404 yields (404, None); other failures raise."""
        last = "no attempt"
        for attempt in range(cfg.MAX_ATTEMPTS):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=cfg.HTTP_TIMEOUT_SEC)
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == cfg.HTTP_NOT_FOUND:
                    return cfg.HTTP_NOT_FOUND, None
                if resp.status_code == cfg.HTTP_OK:
                    return cfg.HTTP_OK, resp.json()
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in cfg.RETRY_STATUSES:
                    raise SecFetchError(f"{url}: {last}")
            delay = min(cfg.BACKOFF_BASE_SEC * 2**attempt, cfg.BACKOFF_CAP_SEC)
            LOG.warning("SEC fetch %s failed (%s); retry in %.0fs", url, last, delay)
            self._sleep(delay)
        raise SecFetchError(f"{url}: gave up after {cfg.MAX_ATTEMPTS} attempts ({last})")
