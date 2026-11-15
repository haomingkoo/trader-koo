from __future__ import annotations

import datetime as dt
import json
import re
import time
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


def test_write_reports_publishes_latest_copies_by_default(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b"  # parents are created
    report = {"generated_ts": "2026-10-25T00:00:00Z", "meta": {}}

    paths = write_reports(report, out)

    json_text = json.dumps(report, separators=(",", ":")) + "\n"
    md_text = (out / "daily_report_20261025T000000Z.md").read_text(encoding="utf-8")
    # to_markdown() ends its last section with a newline; write_reports adds exactly one more.
    assert md_text.startswith("# Trader Koo Daily Report\n") and md_text.endswith("\n- no log lines\n\n")
    assert paths == {
        "json_path": str(out / "daily_report_20261025T000000Z.json"),
        "md_path": str(out / "daily_report_20261025T000000Z.md"),
        "latest_json": str(out / "daily_report_latest.json"),
        "latest_md": str(out / "daily_report_latest.md"),
        "retained_snapshots": 1,
        "pruned_snapshots": 0,
        "pruned_files": 0,
    }
    for name, text in (("daily_report_20261025T000000Z.json", json_text), ("daily_report_20261025T000000Z.md", md_text),
                       ("daily_report_latest.json", json_text), ("daily_report_latest.md", md_text)):
        assert (out / name).read_text(encoding="utf-8") == text


def test_write_reports_without_latest_reports_no_latest_paths(tmp_path: Path) -> None:
    paths = write_reports({"generated_ts": "2026-10-25T00:00:00Z"}, tmp_path, run_id="r1", publish_latest=False)

    assert (paths["latest_json"], paths["latest_md"]) == (None, None)
    assert paths["json_path"] == str(tmp_path / "daily_report_20261025T000000Z_r1.json")
    assert not (tmp_path / "daily_report_latest.json").exists()
    assert not (tmp_path / "daily_report_latest.md").exists()


def test_write_reports_counts_pruned_snapshots_and_files(tmp_path: Path) -> None:
    for suffix in (".json", ".md"):
        (tmp_path / f"daily_report_20200101T000000Z{suffix}").write_text("{}")

    paths = write_reports({"generated_ts": "2026-10-25T00:00:00Z"}, tmp_path, publish_latest=False)

    assert (paths["retained_snapshots"], paths["pruned_snapshots"], paths["pruned_files"]) == (1, 1, 2)


def test_write_reports_stamps_the_run_id_into_report_meta(tmp_path: Path) -> None:
    replaced = {"generated_ts": "2026-10-25T00:00:00Z", "meta": "not a dict"}
    kept = {"generated_ts": "2026-10-25T00:00:00Z", "meta": {"report_run": {"kind": "daily"}, "x": 1}}
    unset = {"generated_ts": "2026-10-25T00:00:00Z", "meta": {"report_run": "bad"}}

    write_reports(replaced, tmp_path, run_id="r1", publish_latest=False)
    write_reports(kept, tmp_path, run_id="r2", publish_latest=False)
    write_reports(unset, tmp_path, run_id="r3", publish_latest=False)

    assert replaced["meta"] == {"report_run": {"run_id": "r1"}}
    assert kept["meta"] == {"report_run": {"kind": "daily", "run_id": "r2"}, "x": 1}
    assert unset["meta"] == {"report_run": {"run_id": "r3"}}
    written = json.loads((tmp_path / "daily_report_20261025T000000Z_r2.json").read_text(encoding="utf-8"))
    assert written["meta"]["report_run"]["run_id"] == "r2"


def test_write_reports_names_an_unparsable_report_by_the_current_utc_time(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TZ", "Asia/Singapore")  # local time must not leak into the name
    time.tzset()
    try:
        paths = write_reports({"generated_ts": "not a timestamp"}, tmp_path, publish_latest=False)
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()

    name = Path(paths["json_path"]).name
    match = re.fullmatch(r"daily_report_(\d{8}T\d{6}Z)\.json", name)
    assert match, name
    stamp = dt.datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    assert abs((dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()) < 600
