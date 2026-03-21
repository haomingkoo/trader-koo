"""Download company logos for all tickers in the price_daily table.

Source: Financial Modeling Prep free image endpoint (no API key required).
Logos are cached to disk; existing files are skipped so only new tickers
trigger a download.

CLI usage:
    python -m trader_koo.scripts.cache_logos --db-path /data/trader_koo.db
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

LOG = logging.getLogger("trader_koo.cache_logos")

LOGO_URL_TEMPLATE = "https://financialmodelingprep.com/image-stock/{ticker}.png"
DEFAULT_LOGOS_DIR = Path(os.getenv("TRADER_KOO_LOGOS_DIR", "/data/logos"))
DOWNLOAD_DELAY_SEC = 0.5
REQUEST_TIMEOUT_SEC = 15


def _load_tickers(db_path: Path) -> list[str]:
    """Return sorted list of distinct tickers from price_daily."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT DISTINCT ticker FROM price_daily").fetchall()
        return sorted(
            {str(r[0]).upper().strip() for r in rows if str(r[0] or "").strip()}
        )
    finally:
        conn.close()


def _download_logo(ticker: str, logos_dir: Path) -> bool:
    """Download a single logo. Returns True on success, False on failure."""
    dest = logos_dir / f"{ticker}.png"
    if dest.exists() and dest.stat().st_size > 0:
        return True  # already cached

    url = LOGO_URL_TEMPLATE.format(ticker=ticker)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "trader-koo/1.0"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as resp:
            data = resp.read()
            if len(data) < 100:
                # Likely a placeholder or error page, skip
                LOG.debug("Skipped %s — response too small (%d bytes)", ticker, len(data))
                return False
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
        LOG.debug("Failed to download logo for %s: %s", ticker, exc)
        return False


def cache_logos(db_path: Path, logos_dir: Path) -> dict[str, int]:
    """Download logos for all tickers. Returns stats dict."""
    logos_dir.mkdir(parents=True, exist_ok=True)
    tickers = _load_tickers(db_path)
    total = len(tickers)
    downloaded = 0
    skipped = 0
    failed = 0

    LOG.info("Caching logos for %d tickers → %s", total, logos_dir)

    for idx, ticker in enumerate(tickers, start=1):
        dest = logos_dir / f"{ticker}.png"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            if idx % 100 == 0 or idx == total:
                LOG.info("Progress %d/%d (skipped cached)", idx, total)
            continue

        ok = _download_logo(ticker, logos_dir)
        if ok:
            downloaded += 1
        else:
            failed += 1

        if idx % 50 == 0 or idx == total:
            LOG.info("Downloaded %d/%d logos (new=%d, cached=%d, failed=%d)",
                     idx, total, downloaded, skipped, failed)

        time.sleep(DOWNLOAD_DELAY_SEC)

    LOG.info(
        "Logo caching complete: total=%d, new=%d, cached=%d, failed=%d",
        total, downloaded, skipped, failed,
    )
    return {"total": total, "downloaded": downloaded, "cached": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache company logos for S&P 500 tickers")
    parser.add_argument("--db-path", type=str, default="/data/trader_koo.db",
                        help="Path to SQLite database")
    parser.add_argument("--logos-dir", type=str, default=None,
                        help="Directory to save logos (default: $TRADER_KOO_LOGOS_DIR or /data/logos)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    db_path = Path(args.db_path)
    if not db_path.exists():
        LOG.error("Database not found: %s", db_path)
        raise SystemExit(1)

    logos_dir = Path(args.logos_dir) if args.logos_dir else DEFAULT_LOGOS_DIR
    cache_logos(db_path, logos_dir)


if __name__ == "__main__":
    main()
