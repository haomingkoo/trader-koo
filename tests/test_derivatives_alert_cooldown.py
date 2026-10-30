"""Crypto derivatives alerts: the cooldown survives a restart and is spent only by a sent alert."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

from trader_koo.crypto import derivatives

# Shape of a derivatives summary as built by the collector (values from the 2026-10-30 BTC alert).
SUMMARY = {"funding": [], "fng": None,
           "ls_ratios": [{"symbol": "BTC-USD", "ratio": 2.72, "long_pct": 73.1, "short_pct": 26.9}]}


def _run(db: Path, sent: bool) -> int:
    with patch("trader_koo.notifications.telegram.is_configured", return_value=True), \
         patch("trader_koo.notifications.telegram.send_message", return_value=sent):
        return derivatives.check_and_alert(db, SUMMARY)


def test_cooldown_survives_a_process_restart(tmp_path: Path):
    db = tmp_path / "t.db"
    sqlite3.connect(db).close()
    assert _run(db, sent=True) == 1
    assert not hasattr(derivatives, "_DERIV_ALERT_COOLDOWN")  # no process-local state left to lose
    assert _run(db, sent=True) == 0  # a restart must not resend the same extreme within 6 hours


def test_a_failed_send_does_not_spend_the_cooldown(tmp_path: Path):
    db = tmp_path / "t.db"
    sqlite3.connect(db).close()
    assert _run(db, sent=False) == 0
    assert _run(db, sent=True) == 1
