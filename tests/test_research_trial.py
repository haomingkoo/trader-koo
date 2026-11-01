"""Trial recording: seal policy, seeds, lineage stamp and entry format, and replay of every recorded run event.

The pinned ids were captured from the runners as they were before ``trial.py`` existed (commit 6527de68), with the
clock and code version frozen. They prove the refactor writes the same ledger; they say nothing about markets.
A change to config/research.toml that changes a config value changes these ids too, by design.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_research_bench import _build_db, _fixture_frames
from tests.test_research_v2 import _db as _v2_db
from tests.test_research_v3 import _db as _v3_db
from trader_koo.research import macro
from trader_koo.research.bench import ledger, trial, v2_run, v3_run, v7_rules, v7_run
from trader_koo.research.bench.__main__ import main as bench_main
from trader_koo.research.bench.data import load_market
from trader_koo.research.bench.run import BenchConfig, run_bench
from trader_koo.research.bench.v3_data import load_frame

DOCS = Path(__file__).resolve().parents[1] / "docs" / "research"
FIX = Path(__file__).parent / "fixtures" / "research_bench"
FROZEN_AT = "2026-01-02T03:04:05+00:00"
FROZEN_CODE = "golden0000000000000000000000000000000000"
RESULT_KEYS = ("summary", "spy", "settings_sha256")  # numbers and the settings hash are not part of the id contract


def _freeze(monkeypatch, at: str = FROZEN_AT, code: str = FROZEN_CODE) -> None:
    fixed = datetime.fromisoformat(at)

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    monkeypatch.setattr(trial, "datetime", Frozen)
    monkeypatch.setattr(ledger, "datetime", Frozen)
    monkeypatch.setattr(ledger, "code_version", lambda: code)


def _lineage_hash(entry: dict) -> str:
    keep = {k: v for k, v in entry.items() if k not in RESULT_KEYS}
    return hashlib.sha256(ledger.canonical(keep).encode()).hexdigest()[:16]


def _shape(path: Path) -> list[tuple]:
    return [(e["event"], e.get("trial_id"), e.get("seed"), _lineage_hash(e)) for e in ledger.read(path)]


# ---- every committed ledger entry re-records byte for byte -----------------------------------------------------------

def _historical_entries() -> list[tuple[str, dict]]:
    return [(f"{p.name}:{e['trial_id']}", e) for p in sorted(DOCS.glob("ledger-*.jsonl"))
            for e in ledger.read(p) if "trial_id" in e]


@pytest.mark.parametrize("name,entry", _historical_entries(), ids=lambda x: x if isinstance(x, str) else "")
def test_committed_entry_is_reproduced_through_the_trial_interface(name, entry, tmp_path, monkeypatch):
    _freeze(monkeypatch, entry["timestamp"], entry["code_version"])
    t = trial.Trial(tmp_path / "l.jsonl", entry["sealed_boundary"])
    assert t.configure(entry["candidate"], entry["config"]) == entry["seed"]
    params = entry["config"]["params"] if entry["event"] == "run" else entry["config"]
    core = set(trial.CORE_FIELDS) | {"settings_sha256"}
    extra = {k: v for k, v in entry.items() if k not in core}
    got_id = t.record(entry["event"], entry["summary"], params=params, sealed_opened=entry["sealed_opened"],
                      sealed_boundary=entry["sealed_boundary"], **extra)
    written = ledger.read(t.path)[-1]
    assert got_id == entry["trial_id"]
    strip = lambda e: {k: v for k, v in e.items() if k != "settings_sha256"}  # noqa: E731
    assert ledger.canonical(strip(written)) == ledger.canonical(strip(entry))


def test_committed_ledgers_cover_every_recorded_event_name():
    assert {e["event"] for _, e in _historical_entries()} >= {"run", "event_study_run", "exposure_run"}


# ---- each runner writes what it wrote before the refactor ------------------------------------------------------------

SEAL_PINNED = ("seal", None, None, "33f77731bf53229d")


@pytest.fixture(scope="module")
def bench_market(tmp_path_factory):
    return load_market(_build_db(str(tmp_path_factory.mktemp("trial") / "b.db"), *_fixture_frames()))


@pytest.mark.parametrize("label,make,expected", [
    ("dev", lambda m: BenchConfig.from_settings("momentum_12_1", n_random=5, start=m.sessions[240]),
     [SEAL_PINNED, ("run", "6a6d7b91aca5", 2172845442, "03fc5ba5f245287c")]),
    ("sealed", lambda m: BenchConfig.from_settings("equal_weight", n_random=5, open_sealed=True),
     [SEAL_PINNED, ("sealed_open", None, 129809638, "16a3800977ff0c6e"),
      ("run", "2b909fbcac9a", 129809638, "7b54e662d453ef33")]),
    ("params", lambda m: BenchConfig.from_settings("low_volatility", n_random=3, params={"lookback": 40}),
     [SEAL_PINNED, ("run", "df7934c632f3", 934454841, "07b52c01505635eb")]),
])
def test_bench_run_entries_match_the_pre_refactor_capture(label, make, expected, bench_market, tmp_path, monkeypatch):
    _freeze(monkeypatch)
    path = tmp_path / f"{label}.jsonl"
    out = run_bench(bench_market, make(bench_market), path)
    assert _shape(path) == expected and out["seed"] == expected[-1][2]


def test_v2_entry_matches_the_pre_refactor_capture(tmp_path, monkeypatch):
    _freeze(monkeypatch)
    path = tmp_path / "v2.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2019-01-02"})
    out = v2_run.run_v2(_v2_db(tmp_path / "v2.db"), path)
    assert _shape(path)[1] == ("event_study_run", "ddef0a60a357", 729545597, "3048b7f76c675393")
    assert out["trial_id"] == "ddef0a60a357" and out["seed"] == 729545597


def test_v3_entry_matches_the_pre_refactor_capture(tmp_path, monkeypatch):
    _freeze(monkeypatch)
    path = tmp_path / "v3.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    out = v3_run.run_v3("unused.db", path, load_frame(_v3_db(tmp_path / "v3.db"), "2012-10-01"))
    assert _shape(path)[1] == ("event_study_run", "fecb561c5772", 3223540573, "539b2d581c09f861")
    assert out["trial_id"] == "fecb561c5772" and out["seed"] == 3223540573


def _v7_inputs(db: str, boundary: str) -> v7_rules.Inputs:
    """Real SPY and ^VIX bars from the v3 fixture; macro tables empty, so every macro rule holds at 1."""
    bars = pd.read_csv(FIX / "index_bars.csv.gz")
    spy = bars[(bars.ticker == "SPY") & (bars.date < boundary)].sort_values("date")
    vix = bars[bars.ticker == "^VIX"].set_index("date")["close"]
    dates = list(spy.date)
    conn = sqlite3.connect(":memory:")
    macro.ensure_schema(conn)
    close = spy.close.to_numpy(float)
    return v7_rules.Inputs(dates, close, spy.open.to_numpy(float), close.copy(), spy.volume.to_numpy(float),
                           vix.reindex(dates).to_numpy(float), np.zeros(len(dates)), conn)


@pytest.fixture
def v7_fixture(monkeypatch):
    monkeypatch.setattr(v7_run, "load_inputs", _v7_inputs)
    monkeypatch.setattr(v7_run, "DEV_START", "2010-06-01")


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_v7_entry_matches_the_pre_refactor_capture(tmp_path, monkeypatch, v7_fixture):
    _freeze(monkeypatch)
    path = tmp_path / "v7.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    out = v7_run.run_v7("unused.db", path)
    assert _shape(path)[1] == ("exposure_run", "5fd20710820a", 2956236113, "16d958fbfac20849")
    assert out["trial_id"] == "5fd20710820a"


# ---- the interface ---------------------------------------------------------------------------------------------------

def test_seeds_follow_the_recorded_formulas(tmp_path):
    t = trial.Trial(tmp_path / "l.jsonl", "2020-01-02")
    config = {"family": "x", "boundary": "2020-01-02"}
    assert t.configure("c", config) == ledger.seed_from_config(config)
    assert t.seed_for("S1") == ledger.seed_from_config({**config, "spec": "S1"})
    assert t.seed_for("H3", field="rule") == ledger.seed_from_config({**config, "rule": "H3"})


def test_stamp_is_taken_once_and_shared_by_every_entry_of_the_trial(tmp_path, monkeypatch):
    _freeze(monkeypatch)
    t = trial.Trial(tmp_path / "l.jsonl", "2020-01-02")
    t.configure("c", {"a": 1})
    first = t.stamp()
    _freeze(monkeypatch, at="2030-01-01T00:00:00+00:00", code="other")
    t.open_sealed()
    trial_id = t.record("run", {}, params={})
    opened, run = ledger.read(t.path)
    assert t.stamp() is first and opened["timestamp"] == run["timestamp"] == FROZEN_AT
    assert trial_id == run["trial_id"] == ledger.short_id(first) and opened["prior_openings"] == 0


def test_record_refuses_a_field_the_trial_owns(tmp_path):
    t = trial.Trial(tmp_path / "l.jsonl", "2020-01-02")
    t.configure("c", {"a": 1})
    with pytest.raises(ValueError, match="set by the trial"):
        t.record("run", {}, params={}, trial_id="forged")
    assert not t.path.exists()


def test_unconfigured_trial_has_no_seed_or_stamp(tmp_path):
    t = trial.Trial(tmp_path / "l.jsonl", "2020-01-02")
    with pytest.raises(RuntimeError):
        t.seed_for("S1")
    with pytest.raises(RuntimeError):
        t.stamp()


# ---- seal policy -----------------------------------------------------------------------------------------------------

SESSIONS = [f"2020-01-{d:02d}" for d in range(1, 11)]


def test_pin_if_missing_writes_the_seal_once_and_reads_it_back(tmp_path):
    path = tmp_path / "l.jsonl"
    first = trial.start(path, trial.PIN_IF_MISSING, SESSIONS).boundary
    assert first == SESSIONS[ledger.seal_boundary(len(SESSIONS))]
    again = trial.start(path, trial.PIN_IF_MISSING, SESSIONS + ["2020-01-11", "2020-01-12"]).boundary
    assert again == first and [e["event"] for e in ledger.read(path)] == ["seal"]


def test_read_only_never_writes_a_seal(tmp_path):
    path = tmp_path / "l.jsonl"
    with pytest.raises(LookupError, match="no seal entry"):
        trial.start(path, trial.READ_ONLY)
    assert not path.exists()
    ledger.append(path, {"event": "seal", "boundary_date": "2020-01-08"})
    assert trial.start(path, trial.READ_ONLY).boundary == "2020-01-08"
    assert v3_run.sealed_boundary(path) == "2020-01-08" and len(ledger.read(path)) == 1


def test_seal_policy_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="unknown seal policy"):
        trial.start(tmp_path / "l.jsonl", "pin")
    with pytest.raises(ValueError, match="needs the sessions"):
        trial.start(tmp_path / "l.jsonl", trial.PIN_IF_MISSING)


# ---- replay of each recorded run event -------------------------------------------------------------------------------

def _last_trial(path: Path) -> str:
    return ledger.read(path)[-1]["trial_id"]


def test_replay_matches_a_v3_event_study(tmp_path, capsys):
    path, db = tmp_path / "l.jsonl", _v3_db(tmp_path / "e.db")
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    assert bench_main(["events", "--db", db, "--family", "v3", "--ledger", str(path)]) == 0
    tid = _last_trial(path)
    capsys.readouterr()
    assert bench_main(["replay", "--db", db, "--trial", tid, "--ledger", str(path)]) == 0
    assert f"MATCHES ledger entry {tid}" in capsys.readouterr().out and len(ledger.read(path)) == 3


def test_replay_matches_a_v2_event_study(tmp_path, capsys):
    path, db = tmp_path / "l.jsonl", _v2_db(tmp_path / "e.db")
    ledger.append(path, {"event": "seal", "boundary_date": "2019-01-02"})
    tid = v2_run.run_v2(db, path)["trial_id"]
    assert bench_main(["replay", "--db", db, "--trial", tid, "--ledger", str(path)]) == 0
    assert "MATCHES" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_replay_matches_a_v7_exposure_run(tmp_path, capsys, v7_fixture):
    path = tmp_path / "l.jsonl"
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    tid = v7_run.run_v7("unused.db", path)["trial_id"]
    assert bench_main(["replay", "--db", "unused.db", "--trial", tid, "--ledger", str(path)]) == 0
    assert "MATCHES" in capsys.readouterr().out


def test_replay_reports_a_study_whose_config_no_longer_matches(tmp_path, capsys):
    path, db = tmp_path / "l.jsonl", _v3_db(tmp_path / "e.db")
    ledger.append(path, {"event": "seal", "boundary_date": "2012-10-01"})
    v3_run.run_v3(db, path)
    entry = ledger.read(path)[-1]
    tampered = {**entry, "trial_id": "tampered0000", "config": {**entry["config"], "sessions": 1}}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(ledger.canonical(tampered) + "\n")
    assert bench_main(["replay", "--db", db, "--trial", "tampered0000", "--ledger", str(path)]) == 1
    assert "different config" in capsys.readouterr().out


def test_replay_refuses_a_study_family_it_cannot_rerun(tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    ledger.append(path, {"event": "exposure_run", "family": "v10", "trial_id": "abc", "config": {}, "summary": {}})
    assert bench_main(["replay", "--db", "unused.db", "--trial", "abc", "--ledger", str(path)]) == 2
    assert "not supported" in capsys.readouterr().err


def test_replay_still_ignores_events_that_are_not_runs(tmp_path, capsys):
    path = tmp_path / "l.jsonl"
    ledger.append(path, {"event": "portfolio_study_run", "family": "v9", "trial_id": "abc"})
    assert bench_main(["replay", "--db", "unused.db", "--trial", "abc", "--ledger", str(path)]) == 2
    assert "no run entry" in capsys.readouterr().err
