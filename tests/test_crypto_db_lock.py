"""Crypto bar persistence while another writer holds the SQLite lock.

On 2026-11-14 03:09-03:10 UTC the report held the write lock, the flush
dropped bars with "database is locked", and the Binance WS thread, blocked in
a synchronous candle write, missed its ping/pong deadline.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
import time
from pathlib import Path

import pytest

from trader_koo.crypto import service, storage
from trader_koo.crypto.binance_ws import BinanceWSClient
from trader_koo.crypto.models import CryptoBar


# Relative to now: the flush prunes 1m bars older than 30 days.
_HOUR = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=1)


def _ts(minute: int) -> str:
    return (_HOUR + dt.timedelta(minutes=minute)).isoformat()


def _bar(minute: int, interval: str = "1m") -> CryptoBar:
    return CryptoBar(
        symbol="BTC-USD",
        timestamp=_HOUR + dt.timedelta(minutes=minute),
        interval=interval,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=3.0,
    )


def _stored(db_path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT timestamp, interval FROM crypto_bars ORDER BY timestamp, interval"
        ).fetchall()
    finally:
        conn.close()


@pytest.fixture()
def crypto_db(tmp_path: Path, monkeypatch) -> Path:
    db_path = tmp_path / "crypto.db"
    conn = sqlite3.connect(db_path)
    storage.ensure_crypto_schema(conn)
    conn.close()
    client = BinanceWSClient()  # never started: no network
    monkeypatch.setattr(service, "_client", client)
    monkeypatch.setattr(service, "_db_path_str", str(db_path))
    # Keep the lock wait short so the test runs in well under a second.
    monkeypatch.setattr(storage, "CRYPTO_DB_BUSY_TIMEOUT_SEC", 0.2, raising=False)
    return db_path


@pytest.fixture()
def write_lock(crypto_db: Path):
    """Hold the database write lock the way a long report transaction does."""
    holder = sqlite3.connect(crypto_db, isolation_level=None)
    holder.execute("BEGIN EXCLUSIVE")
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            holder.execute("COMMIT")
            holder.close()
            released = True

    yield release
    release()


def test_locked_flush_keeps_bars_and_writes_them_on_the_next_flush(crypto_db, write_lock, caplog):
    service._client._pending_bars.extend([_bar(9), _bar(10)])

    with caplog.at_level(logging.WARNING, logger="trader_koo.crypto.service"):
        service._flush_bars_to_db()
    write_lock()

    assert _stored(crypto_db) == []

    service._flush_bars_to_db()

    assert _stored(crypto_db) == [(_ts(9), "1m"), (_ts(10), "1m")]
    assert any("retained 2 crypto bars" in r.getMessage() for r in caplog.records)


def test_finalized_candle_does_not_touch_the_db_on_the_ws_thread(crypto_db, write_lock):
    started = time.monotonic()
    service._on_candle_finalized(_bar(5, interval="5m"))
    elapsed = time.monotonic() - started
    write_lock()

    # The WS thread must return at once; a synchronous write waits out the lock.
    assert elapsed < 0.1, f"WS callback blocked for {elapsed:.2f}s"

    service._flush_bars_to_db()

    assert _stored(crypto_db) == [(_ts(5), "5m")]


def test_crypto_connection_waits_30s_for_locks_in_wal_mode(tmp_path: Path):
    conn = storage.connect_crypto_db(str(tmp_path / "wal.db"))
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_non_lock_error_drops_the_batch_loudly(tmp_path: Path, monkeypatch, caplog):
    empty_db = tmp_path / "no-schema.db"
    sqlite3.connect(empty_db).close()
    monkeypatch.setattr(service, "_client", BinanceWSClient())
    monkeypatch.setattr(service, "_db_path_str", str(empty_db))
    service._client._pending_bars.append(_bar(1))

    with caplog.at_level(logging.ERROR, logger="trader_koo.crypto.service"):
        service._flush_bars_to_db()

    assert service._client.drain_pending_bars() == []
    assert any("dropped 1 crypto bars" in r.getMessage() for r in caplog.records)
