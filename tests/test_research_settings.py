"""The settings loader fails loudly; the shipped file is complete and unchanged in value."""
from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path

import pytest

from trader_koo.research import settings as st
from trader_koo.research.bench import ledger


def _text() -> str:
    return st.DEFAULT_PATH.read_text(encoding="utf-8")


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "research.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_file_loads_and_hashes_its_own_bytes():
    s = st.load_settings()
    assert s.sha256 == hashlib.sha256(st.DEFAULT_PATH.read_bytes()).hexdigest()
    assert s.bench.cost_bps == 10.0 and s.bench.top_n == 20 and s.bench.universe == "sp500_pit"
    assert s.collection.universe_target_days == (30, 60, 90)
    assert st.get() is st.get()  # loaded once


def test_every_entry_carries_a_unit_and_a_reason():
    entries = re.findall(r"^\w+ = \{ value = .*\}$", _text(), flags=re.M)
    sections = [f for f in dataclasses.fields(st.ResearchSettings) if f.name not in ("sha256", "path")]
    assert len(entries) == sum(len(dataclasses.fields(f.type if not isinstance(f.type, str) else getattr(st, f.type))) for f in sections)
    assert all('unit = "' in e and 'reason = "' in e for e in entries)


def test_missing_key_fails(tmp_path):
    text = re.sub(r"^cost_bps = .*\n", "", _text(), flags=re.M)
    with pytest.raises(st.SettingsError, match=r"missing keys \['cost_bps'\]"):
        st.load_settings(_write(tmp_path, text))


def test_unknown_key_fails(tmp_path):
    text = _text().replace("[quality]\n", '[quality]\nsurprise = { value = 1, unit = "u", reason = "r" }\n')
    with pytest.raises(st.SettingsError, match="unknown keys"):
        st.load_settings(_write(tmp_path, text))


def test_missing_and_unknown_sections_fail(tmp_path):
    text = _text().replace("[membership]", "[membership_renamed]")
    with pytest.raises(st.SettingsError, match="membership"):
        st.load_settings(_write(tmp_path, text))
    with pytest.raises(st.SettingsError, match="unknown sections"):
        st.load_settings(_write(tmp_path, _text() + '\n[extra]\nx = { value = 1, unit = "u", reason = "r" }\n'))


@pytest.mark.parametrize("old, new", [
    ('n_random = { value = 200,', 'n_random = { value = "200",'),      # str for int
    ('n_random = { value = 200,', 'n_random = { value = 200.5,'),      # float for int
    ('n_random = { value = 200,', 'n_random = { value = true,'),       # bool is not an int
    ('cost_bps = { value = 10.0,', 'cost_bps = { value = "ten",'),     # str for float
    ('benchmark_ticker = { value = "SPY",', 'benchmark_ticker = { value = 5,'),
    ('universe_target_days = { value = [30, 60, 90],', 'universe_target_days = { value = [30, "60", 90],'),
    ('universe_target_days = { value = [30, 60, 90],', 'universe_target_days = { value = 30,'),
])
def test_wrong_type_fails(tmp_path, old, new):
    assert old in _text()
    with pytest.raises(st.SettingsError):
        st.load_settings(_write(tmp_path, _text().replace(old, new, 1)))


def test_an_integer_is_accepted_for_a_float_setting(tmp_path):
    text = _text().replace("cost_bps = { value = 10.0,", "cost_bps = { value = 10,", 1)
    assert st.load_settings(_write(tmp_path, text)).bench.cost_bps == 10.0


def test_unit_and_reason_are_required(tmp_path):
    no_reason = re.sub(r'^(cost_bps = \{ value = 10.0, unit = "[^"]*"), reason = "[^"]*" \}$', r"\1 }", _text(), flags=re.M)
    with pytest.raises(st.SettingsError, match="exactly value, unit and reason"):
        st.load_settings(_write(tmp_path, no_reason))
    blank = re.sub(r'^(cost_bps = \{ value = 10.0, unit = "[^"]*", reason = )"[^"]*"', r'\1" "', _text(), flags=re.M)
    with pytest.raises(st.SettingsError, match="reason must be a non-empty string"):
        st.load_settings(_write(tmp_path, blank))
    bare = re.sub(r"^cost_bps = .*$", "cost_bps = 10.0", _text(), flags=re.M)
    with pytest.raises(st.SettingsError, match="exactly value, unit and reason"):
        st.load_settings(_write(tmp_path, bare))


def test_missing_file_and_invalid_toml_fail(tmp_path):
    with pytest.raises(st.SettingsError, match="cannot read"):
        st.load_settings(tmp_path / "absent.toml")
    with pytest.raises(st.SettingsError, match="not valid TOML"):
        st.load_settings(_write(tmp_path, "[bench\n"))


def test_cross_checks_reject_inconsistent_values(tmp_path):
    bad_seal = _text().replace("seal_fraction = { value = 0.2,", "seal_fraction = { value = 1.5,", 1)
    with pytest.raises(st.SettingsError, match="seal_fraction"):
        st.load_settings(_write(tmp_path, bad_seal))
    bad_cron = _text().replace("universe_cron_minute_utc = { value = 45,", "universe_cron_minute_utc = { value = 50,", 1)
    with pytest.raises(st.SettingsError, match="cron time"):
        st.load_settings(_write(tmp_path, bad_cron))


def test_values_match_the_registered_rules_and_the_old_constants():
    """The values that moved out of code are the ones that were in code (replay depends on it)."""
    s = st.load_settings()
    b = s.bench
    assert (b.cost_bps, b.stress_bps, b.n_random, b.top_n, b.buffer_n, b.review_every, b.min_hold) == (
        10.0, 20.0, 200, 20, 40, 10, 20)  # docs/research/hypotheses-v1.md common rules
    assert b.seal_fraction == 0.2 and b.trading_days_per_year == 252 and b.summary_digits == 10
    q = s.quality
    assert (q.zero_volume_run_sessions, q.seam_move_fraction, q.erratic_move_fraction, q.erratic_moves_per_year) == (
        5, 0.25, 0.25, 7)
    assert (s.total_return.max_abs_daily_move, s.total_return.max_dividend_fraction) == (0.40, 0.10)


def test_ledger_entries_record_the_settings_hash(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger.append(path, {"event": "x"})
    ledger.append(path, {"event": "y"})
    assert [e["settings_sha256"] for e in ledger.read(path)] == [st.get().sha256] * 2


def test_code_version_fails_instead_of_returning_a_placeholder(monkeypatch):
    def broken(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(ledger.subprocess, "run", broken)
    with pytest.raises(ledger.CodeVersionError):
        ledger.code_version()
