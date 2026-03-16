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

    def test_scheduler_not_running_on_create(self):
        scheduler = create_scheduler()

        assert not scheduler.running


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
