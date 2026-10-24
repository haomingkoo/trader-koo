from __future__ import annotations

from pathlib import Path

from trader_koo.report.serializer import write_reports


def test_write_reports_keeps_an_old_run_artifact_with_a_non_uuid_run_id(tmp_path: Path) -> None:
    report = {"generated_ts": "2020-01-01T00:00:00Z", "meta": {}}

    paths = write_reports(report, tmp_path, run_id="fixture-run", publish_latest=False)

    assert Path(paths["json_path"]).exists()


def test_write_reports_prunes_an_old_unmanaged_snapshot(tmp_path: Path) -> None:
    stale = tmp_path / "daily_report_20200101T000000Z.json"
    stale.write_text("{}")

    write_reports({"generated_ts": "2026-10-25T00:00:00Z", "meta": {}}, tmp_path, publish_latest=False)

    assert not stale.exists()
