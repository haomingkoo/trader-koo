"""run_yolo_patterns.main(): CLI options, passes, retention and exit status.

Only the model constructor and the per-ticker pass are stubbed; argument
parsing, the DB setup, retention pruning and the exit code run for real.
"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trader_koo.scripts import run_yolo_patterns as yolo


class FakeModel:
    def __init__(self) -> None:
        self.overrides = {"imgsz": 0, "conf": 0.0, "iou": 0.0}


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "trader_koo.db"
    sqlite3.connect(db).close()
    return db


def _exit_code() -> int:
    """main()'s process exit status: it returns normally on success."""
    try:
        yolo.main()
    except SystemExit as exc:
        return exc.code
    return 0


def _run(monkeypatch, *argv: str, ok: int = 1, failed: int = 0, model=None) -> SimpleNamespace:
    calls: list[dict] = []
    model = model or FakeModel()

    def fake_pass(passed_model, conn, **kwargs):
        assert passed_model is model
        assert conn.execute("SELECT COUNT(*) FROM yolo_run_events").fetchone() is not None
        calls.append(kwargs)
        return ok, failed, 2

    monkeypatch.setattr(yolo, "load_hf_yolo", lambda: model)
    monkeypatch.setattr(yolo, "_run_pass", fake_pass)
    monkeypatch.setattr(torch, "load", torch.load)  # main() patches torch.load; undo after the test
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", *argv])
    return SimpleNamespace(code=_exit_code(), calls=calls, model=model)


@pytest.mark.parametrize(
    ("ok", "failed", "code"),
    [
        (0, 3, 1),  # every ticker failed
        (5, 0, 0),
        (0, 0, 0),  # nothing needed processing
    ],
)
def test_exit_code_fails_a_run_below_the_coverage_floor(tmp_path, monkeypatch, ok, failed, code) -> None:
    result = _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "daily", ok=ok, failed=failed)

    assert result.code == code


def test_missing_database_exits_before_loading_the_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(yolo, "load_hf_yolo", lambda: pytest.fail("model loaded without a database"))
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", "--db-path", str(tmp_path / "missing.db")])

    code = _exit_code()

    assert code == 1


def test_db_path_defaults_to_the_volume_database(monkeypatch, caplog) -> None:
    monkeypatch.delenv("TRADER_KOO_DB_PATH", raising=False)
    monkeypatch.setattr(yolo, "load_hf_yolo", lambda: pytest.fail("model loaded without a database"))
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns"])
    if Path("/data/trader_koo.db").exists():
        pytest.skip("a real volume database exists on this machine")

    _exit_code()

    assert "DB not found at /data/trader_koo.db" in caplog.text


def test_db_path_defaults_to_the_env_setting(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRADER_KOO_DB_PATH", str(_db(tmp_path)))

    assert _run(monkeypatch, "--timeframe", "daily").code == 0


def test_defaults_run_daily_then_weekly_with_default_options(tmp_path, monkeypatch) -> None:
    result = _run(monkeypatch, "--db-path", str(_db(tmp_path)))

    shared = {
        "only_new": False,
        "sleep_s": 0.05,
        "dpi": yolo.DEFAULT_DPI,
        "fig_w": yolo.DEFAULT_FIG_W,
        "fig_h": yolo.DEFAULT_FIG_H,
        "max_secs_per_ticker": yolo.DEFAULT_MAX_SECS_PER_TICKER,
    }
    run_id = result.calls[0]["run_id"]
    assert re.fullmatch(r"\d{8}T\d{6}Z-yolo-\d+", run_id)
    assert result.calls == [
        {"run_id": run_id, "timeframe": "daily", "lookback_days": yolo.DEFAULT_LOOKBACK_DAYS, **shared},
        {"run_id": run_id, "timeframe": "weekly", "lookback_days": yolo.DEFAULT_WEEKLY_LOOKBACK_DAYS, **shared},
    ]
    assert result.model.overrides == {
        "imgsz": yolo.DEFAULT_IMGSZ,
        "conf": yolo.DEFAULT_CONF,
        "iou": yolo.DEFAULT_IOU,
        "agnostic_nms": False,
        "max_det": 1000,
    }


def test_cli_options_reach_the_pass_and_the_model(tmp_path, monkeypatch) -> None:
    result = _run(
        monkeypatch,
        "--db-path", str(_db(tmp_path)),
        "--timeframe", "weekly",
        "--lookback-days", "11",
        "--weekly-lookback-days", "22",
        "--only-new",
        "--sleep", "0",
        "--dpi", "33",
        "--fig-w", "4.5",
        "--fig-h", "2.5",
        "--imgsz", "320",
        "--conf", "0.5",
        "--iou", "0.6",
        "--max-seconds-per-ticker", "9",
        "--run-id", "  run-7  ",
    )

    assert result.calls == [{
        "run_id": "run-7",
        "timeframe": "weekly",
        "lookback_days": 22,
        "only_new": True,
        "sleep_s": 0.0,
        "dpi": 33,
        "fig_w": 4.5,
        "fig_h": 2.5,
        "max_secs_per_ticker": 9.0,
    }]
    assert result.model.overrides == {"imgsz": 320, "conf": 0.5, "iou": 0.6, "agnostic_nms": False, "max_det": 1000}


def test_daily_timeframe_runs_only_the_daily_pass(tmp_path, monkeypatch) -> None:
    result = _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "daily", "--lookback-days", "11")

    assert [(c["timeframe"], c["lookback_days"]) for c in result.calls] == [("daily", 11)]


def test_retention_keeps_the_configured_as_of_dates_per_timeframe(tmp_path, monkeypatch) -> None:
    db = _db(tmp_path)
    keep = {"daily": yolo.DEFAULT_DAILY_KEEP_ASOFS, "weekly": yolo.DEFAULT_WEEKLY_KEEP_ASOFS}
    with sqlite3.connect(db) as conn:
        yolo.ensure_yolo_table(conn)
        for timeframe, keep_n in keep.items():
            conn.executemany(
                "INSERT INTO yolo_patterns (ticker,timeframe,pattern,confidence,x0_date,x1_date,"
                "y0,y1,lookback_days,as_of_date) VALUES ('AAA',?,'W_Bottom',0.9,'a','b',1,2,10,?)",
                [(timeframe, f"as-of-{i:04d}") for i in range(keep_n + 1)],
            )

    assert _run(monkeypatch, "--db-path", str(db)).code == 0

    with sqlite3.connect(db) as conn:
        retained = dict(conn.execute(
            "SELECT timeframe, COUNT(DISTINCT as_of_date) FROM yolo_patterns GROUP BY timeframe"
        ).fetchall())
        oldest = conn.execute("SELECT MIN(as_of_date) FROM yolo_patterns WHERE timeframe='daily'").fetchone()[0]
    assert retained == keep
    assert oldest == "as-of-0001"


def test_model_load_failure_exits_non_zero(tmp_path, monkeypatch) -> None:
    def broken():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(yolo, "load_hf_yolo", broken)
    monkeypatch.setattr(yolo, "_run_pass", lambda *a, **k: pytest.fail("ran without a model"))
    monkeypatch.setattr(torch, "load", torch.load)
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", "--db-path", str(_db(tmp_path))])

    code = _exit_code()

    assert code == 1


def test_model_init_timeout_exits_non_zero_and_bounds_hub_downloads(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.setattr(yolo, "load_hf_yolo", lambda: time.sleep(5))
    monkeypatch.setattr(torch, "load", torch.load)
    monkeypatch.setattr(sys, "argv", [
        "run_yolo_patterns", "--db-path", str(_db(tmp_path)), "--model-init-timeout-sec", "1.0",
    ])

    started = time.monotonic()
    code = _exit_code()

    assert code == 1
    assert time.monotonic() - started < 4
    assert yolo.os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "1"


def test_zero_init_timeout_leaves_hub_timeout_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)

    _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "daily", "--model-init-timeout-sec", "0")

    assert "HF_HUB_DOWNLOAD_TIMEOUT" not in yolo.os.environ


def test_legacy_weights_load_without_weights_only_unless_asked(tmp_path, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: seen.append((a, kw)))
    monkeypatch.setattr(yolo, "_run_pass", lambda *a, **k: (1, 0, 0))

    def load_model():
        torch.load("weights.pt")
        torch.load("weights.pt", weights_only=True)
        return FakeModel()

    monkeypatch.setattr(yolo, "load_hf_yolo", load_model)
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", "--db-path", str(_db(tmp_path)), "--timeframe", "daily"])
    _exit_code()

    assert seen == [(("weights.pt",), {"weights_only": False}), (("weights.pt",), {"weights_only": True})]


def test_timeframe_must_be_daily_weekly_or_both(tmp_path, monkeypatch) -> None:
    result = _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "both")
    assert [c["timeframe"] for c in result.calls] == ["daily", "weekly"]

    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", "--db-path", str(_db(tmp_path)), "--timeframe", "hourly"])
    code = _exit_code()
    assert code == 2


def test_exit_verdict_uses_totals_across_both_passes(tmp_path, monkeypatch, caplog) -> None:
    per_pass = iter([(2, 1, 1), (0, 3, 4)])
    monkeypatch.setattr(yolo, "load_hf_yolo", FakeModel)
    monkeypatch.setattr(yolo, "_run_pass", lambda *a, **k: next(per_pass))
    monkeypatch.setattr(torch, "load", torch.load)
    monkeypatch.setattr(sys, "argv", ["run_yolo_patterns", "--db-path", str(_db(tmp_path)), "--run-id", "r1"])

    code = _exit_code()

    assert code == 1  # 2 of 6 attempted tickers is below the coverage floor
    assert "Done run_id=r1 — ok=2 failed=4 skipped=5" in caplog.text


def test_generated_run_id_uses_utc(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TZ", "Asia/Singapore")
    time.tzset()
    try:
        result = _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "daily")
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()

    stamp = dt.datetime.strptime(result.calls[0]["run_id"][:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    assert abs((dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()) < 600


@pytest.mark.parametrize("volume", [True, False])
def test_model_cache_lives_on_the_volume_only_when_it_exists(tmp_path, monkeypatch, volume: bool) -> None:
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: volume if str(self) == "/data" else real_exists(self))
    monkeypatch.delenv("ULTRALYTICS_CONFIG_DIR", raising=False)

    _run(monkeypatch, "--db-path", str(_db(tmp_path)), "--timeframe", "daily")

    assert yolo.os.environ.get("ULTRALYTICS_CONFIG_DIR") == ("/data/.ultralytics" if volume else None)
