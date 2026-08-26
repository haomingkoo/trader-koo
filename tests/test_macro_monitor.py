from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import patch

from trader_koo.notifications import macro_monitor, telegram


OIL_MOVE = {
    "ticker": "USO",
    "name": "Oil",
    "emoji": "oil",
    "prev_close": 100.0,
    "current": 95.42,
    "change_pct": -4.58,
    "direction": "down",
    "threshold_pct": 3.0,
    "exceeded": True,
}


def test_material_oil_move_has_non_contradictory_mixed_reasoning() -> None:
    quiet_moves = [
        {
            **OIL_MOVE,
            "ticker": ticker,
            "name": name,
            "current": 100.0,
            "change_pct": 0.0,
            "direction": "down",
            "exceeded": False,
        }
        for ticker, name in (
            ("^TNX", "10Y Yield"),
            ("GLD", "Gold"),
            ("^VIX", "VIX"),
        )
    ]
    moves = [OIL_MOVE, *quiet_moves]
    regime = macro_monitor.detect_risk_regime(moves)
    message = macro_monitor._format_macro_alert(moves, regime)

    assert regime["regime"] == "MIXED"
    assert regime["reasoning"] == (
        "Material instrument move; available broader risk signals are not aligned"
    )
    assert "Coverage: 4/8 monitored instruments" in message
    assert "Oil" in message
    assert "-4.58%" in message
    assert "No strong directional signals" not in message


def test_macro_cooldown_is_written_only_after_successful_send(tmp_path) -> None:
    with (
        patch.object(telegram, "is_configured", return_value=True),
        patch.object(telegram, "send_message", side_effect=[False, True]),
        patch.object(macro_monitor, "_ensure_cooldown_table"),
        patch.object(macro_monitor, "_read_last_alert_ts", return_value=0.0),
        patch.object(macro_monitor, "_write_last_alert_ts") as write_cooldown,
        patch.object(macro_monitor, "check_macro_moves", return_value=[OIL_MOVE]),
    ):
        assert macro_monitor.send_macro_alert(tmp_path / "macro.db") is False
        write_cooldown.assert_not_called()

        assert macro_monitor.send_macro_alert(tmp_path / "macro.db") is True

    assert write_cooldown.call_count == 2


def test_same_price_cohort_does_not_repeat_across_utc_midnight_but_higher_band_alerts(
    tmp_path,
) -> None:
    times = iter(
        (
            dt.datetime(2026, 8, 26, 23, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 27, 0, 1, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 8, 27, 1, 2, tzinfo=dt.timezone.utc),
        )
    )

    class _Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            value = next(times)
            return value if tz is not None else value.replace(tzinfo=None)

    severe_oil_move = {
        **OIL_MOVE,
        "current": 93.0,
        "change_pct": -7.0,
    }
    clock_module = SimpleNamespace(datetime=_Clock, timezone=dt.timezone)
    db_path = tmp_path / "macro.db"

    with (
        patch.object(telegram, "is_configured", return_value=True),
        patch.object(telegram, "send_message", return_value=True) as send_message,
        patch.object(
            macro_monitor,
            "check_macro_moves",
            side_effect=[[OIL_MOVE], [OIL_MOVE], [severe_oil_move]],
        ),
        patch.object(macro_monitor, "dt", clock_module),
    ):
        assert macro_monitor.send_macro_alert(db_path) is True
        assert macro_monitor.send_macro_alert(db_path) is False
        assert macro_monitor.send_macro_alert(db_path) is True

    assert send_message.call_count == 2


def test_new_prev_close_cohort_is_a_new_macro_event() -> None:
    regime = macro_monitor.detect_risk_regime([OIL_MOVE])
    next_session_move = {
        **OIL_MOVE,
        "prev_close": 101.0,
        "current": 96.37,
    }

    assert macro_monitor._macro_event_key(
        [OIL_MOVE], regime
    ) != macro_monitor._macro_event_key([next_session_move], regime)
