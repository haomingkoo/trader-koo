"""Unit tests for trader_koo.backend.services.scheduler."""
from __future__ import annotations

import pytest

from trader_koo.backend.services.scheduler import (
    _normalize_update_mode,
    create_scheduler,
)


class TestCreateScheduler:
    def test_returns_background_scheduler(self):
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = create_scheduler()

        assert isinstance(scheduler, BackgroundScheduler)

    def test_has_daily_update_job(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("daily_update")

        assert job is not None

    def test_has_weekly_yolo_job(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("weekly_yolo")

        assert job is not None

    def test_has_options_iv_snapshot_job_by_default(self):
        scheduler = create_scheduler()
        job = scheduler.get_job("options_iv_snapshot")

        assert job is not None

    def test_scheduler_not_running_on_create(self):
        scheduler = create_scheduler()

        assert not scheduler.running

    def test_default_fast_monitor_intervals(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")

        scheduler = create_scheduler()

        assert (
            scheduler.get_job("polymarket_snapshot").trigger.interval.total_seconds()
            == 5 * 60
        )
        assert scheduler.get_job("spike_alerts").trigger.interval.total_seconds() == 5 * 60
        assert scheduler.get_job("macro_alert").trigger.interval.total_seconds() == 10 * 60
        assert (
            scheduler.get_job("hyperliquid_poll").trigger.interval.total_seconds()
            == 5 * 60
        )
        assert (
            scheduler.get_job("site_health_check").trigger.interval.total_seconds()
            == 10 * 60
        )
        assert (
            scheduler.get_job("crypto_health_check").trigger.interval.total_seconds()
            == 15 * 60
        )
        assert (
            scheduler.get_job("derivatives_snapshot").trigger.interval.total_seconds()
            == 15 * 60
        )

    def test_options_iv_snapshot_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("TRADER_KOO_OPTIONS_SNAPSHOT_ENABLED", "0")

        scheduler = create_scheduler()

        assert scheduler.get_job("options_iv_snapshot") is None

    def test_monitor_intervals_can_be_overridden(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TRADER_KOO_POLYMARKET_SNAPSHOT_MINUTES", "7")
        monkeypatch.setenv("TRADER_KOO_SPIKE_ALERT_MINUTES", "11")
        monkeypatch.setenv("TRADER_KOO_HYPERLIQUID_POLL_MINUTES", "13")

        scheduler = create_scheduler()

        assert (
            scheduler.get_job("polymarket_snapshot").trigger.interval.total_seconds()
            == 7 * 60
        )
        assert scheduler.get_job("spike_alerts").trigger.interval.total_seconds() == 11 * 60
        assert (
            scheduler.get_job("hyperliquid_poll").trigger.interval.total_seconds()
            == 13 * 60
        )


class TestNormalizeUpdateMode:
    def test_full_maps_to_full(self):
        assert _normalize_update_mode("full") == "full"

    def test_all_maps_to_full(self):
        assert _normalize_update_mode("all") == "full"

    def test_yolo_maps_to_yolo(self):
        assert _normalize_update_mode("yolo") == "yolo"

    def test_yolo_report_maps_to_yolo(self):
        assert _normalize_update_mode("yolo_report") == "yolo"

    def test_yolo_plus_report_maps_to_yolo(self):
        assert _normalize_update_mode("yolo+report") == "yolo"

    def test_report_maps_to_report(self):
        assert _normalize_update_mode("report") == "report"

    def test_report_only_maps_to_report(self):
        assert _normalize_update_mode("report_only") == "report"

    def test_email_maps_to_report(self):
        assert _normalize_update_mode("email") == "report"

    def test_none_defaults_to_full(self):
        assert _normalize_update_mode(None) == "full"

    def test_unknown_mode_returns_none(self):
        assert _normalize_update_mode("garbage_mode") is None

    def test_case_insensitive(self):
        assert _normalize_update_mode("FULL") == "full"
        assert _normalize_update_mode("Yolo") == "yolo"
        assert _normalize_update_mode("REPORT") == "report"

    def test_whitespace_stripped(self):
        assert _normalize_update_mode("  full  ") == "full"
