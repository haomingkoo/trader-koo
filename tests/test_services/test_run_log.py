"""The cron run log: where it lives, what a line looks like, and how runs are read back."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from trader_koo.backend.services import run_log
from trader_koo.backend.services.run_log import (
    append_run_log,
    last_marker_index,
    latest_run_block,
    marker_rc_ok,
    stage_results,
)
from trader_koo.config import ConfigError

LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00 \[(?P<tag>[^\]]+)\] (?P<msg>.*)$")


def test_log_dir_comes_from_env_and_is_created(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "a" / "b"
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(target))

    assert run_log._resolve_log_dir() == target
    assert target.is_dir()


def test_log_dir_that_already_exists_is_accepted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(tmp_path))

    assert run_log._resolve_log_dir() == tmp_path


def test_log_dir_defaults_when_env_is_unset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRADER_KOO_LOG_DIR", raising=False)
    monkeypatch.setattr(run_log, "DEFAULT_LOG_DIR", tmp_path / "default")

    assert run_log._resolve_log_dir() == tmp_path / "default"


def test_unwritable_log_dir_is_a_config_error_naming_the_path(tmp_path: Path, monkeypatch) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("TRADER_KOO_LOG_DIR", str(blocker / "logs"))

    with pytest.raises(ConfigError, match=f"TRADER_KOO_LOG_DIR is not writable: {re.escape(str(blocker))}"):
        run_log._resolve_log_dir()


def test_append_writes_timestamped_tagged_lines_into_a_new_nested_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "x" / "y"
    path = log_dir / "cron_daily.log"

    append_run_log("SCHED", "first", log_dir=log_dir, run_log_path=path)
    append_run_log("SCHED", "second", log_dir=log_dir, run_log_path=path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [LINE_RE.match(line).group("tag", "msg") for line in lines] == [("SCHED", "first"), ("SCHED", "second")]


def test_append_uses_the_module_log_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(run_log, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(run_log, "RUN_LOG_PATH", tmp_path / "logs" / "cron_daily.log")

    append_run_log("RUN", "hello")

    assert LINE_RE.match((tmp_path / "logs" / "cron_daily.log").read_text(encoding="utf-8").strip())


def test_append_failure_is_logged_not_raised(tmp_path: Path, caplog) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")

    append_run_log("RUN", "lost", log_dir=blocker / "d", run_log_path=blocker / "d" / "log")

    assert "Failed to append run log" in caplog.text


def test_last_marker_index_finds_the_newest_line_with_any_token() -> None:
    lines = ["a", "b token1", "c", "d token2", "e"]

    assert last_marker_index(lines, "token1") == 1
    assert last_marker_index(lines, "token1", "token2") == 3
    assert last_marker_index(lines, "absent") == -1


def test_latest_run_block_starts_at_the_newest_start_marker() -> None:
    lines = ["old", f"{run_log.START_MARKER} mode=full", "x", f"{run_log.START_MARKER} mode=yolo", "y"]

    assert latest_run_block(lines) == lines[3:]
    assert latest_run_block(["a", "b"]) == ["a", "b"]


def test_marker_rc_reads_the_code_after_the_marker() -> None:
    marker = run_log.YOLO_DONE_MARKER

    assert marker_rc_ok(f"ingest rc=2 then {marker} rc=0 sec=3", marker) is True
    assert marker_rc_ok(f"ingest rc=0 then {marker} rc=1 sec=3", marker) is False
    assert marker_rc_ok(f"{marker} no code", marker) is False


def test_stage_marker_on_the_first_line_counts() -> None:
    results = stage_results("full", 0, lambda: ["[INGEST] done rc=0 sec=1", "[YOLO]  Daily pattern detection done. rc=0"])

    assert (results.ingest_ok, results.yolo_ok, results.report_ok) == (True, True, True)


def test_failed_run_does_not_credit_yolo_after_a_failed_ingest() -> None:
    tail = [
        f"{run_log.START_MARKER} mode=full",
        "[INGEST] done rc=2 sec=1",
        "[YOLO]  Daily pattern detection done. rc=0 sec=1",
        "[REPORT] Done. rc=0 sec=1",
    ]

    results = stage_results("full", 1, lambda: tail)

    assert (results.ingest_ok, results.yolo_ok, results.report_ok) == (False, False, False)
