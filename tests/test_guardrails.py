"""Ratcheting guardrails: file size, bare numbers and duplicate-code counts may only fall."""
from __future__ import annotations

from ops.guardrails import check, scan, seed_baseline, update_baseline


def test_new_file_over_limit_fails():
    assert check({"a.py": {"lines": 401, "bare_numbers": 0}}, {}) == ["a.py: 401 lines > 400"]


def test_baselined_file_may_shrink_not_grow():
    base = {"g.py": {"lines": 1700, "bare_numbers": 90}}
    assert check({"g.py": {"lines": 1650, "bare_numbers": 90}}, base) == []
    assert check({"g.py": {"lines": 1701, "bare_numbers": 90}}, base) == ["g.py: 1701 lines > baseline 1700"]


def test_bare_numbers_may_not_exceed_baseline():
    base = {"g.py": {"lines": 1700, "bare_numbers": 90}}
    assert check({"g.py": {"lines": 1700, "bare_numbers": 91}}, base) == ["g.py: 91 bare numbers > baseline 90"]
    assert check({"n.py": {"lines": 10, "bare_numbers": 1}}, {}) == ["n.py: 1 bare numbers > baseline 0"]


def test_small_baselined_file_may_grow_under_the_limit():
    base = {"s.py": {"lines": 120, "bare_numbers": 3}}
    assert check({"s.py": {"lines": 300, "bare_numbers": 3}}, base) == []


def test_renamed_file_keeps_allowance_when_baseline_entry_is_renamed():
    base = {"new/g.py": {"lines": 1700, "bare_numbers": 90}}
    assert check({"new/g.py": {"lines": 1700, "bare_numbers": 90}}, base) == []
    assert check({"new/g.py": {"lines": 1700, "bare_numbers": 90}}, {"old/g.py": base["new/g.py"]}) != []


def test_bare_number_allowlist(tmp_path):
    (tmp_path / "trader_koo").mkdir()
    f = tmp_path / "trader_koo" / "m.py"
    f.write_text("x = 0\ny = 1\nz = -1\nw = 2\nv = 100\nthreshold = 0.35\n")
    assert scan(tmp_path)["trader_koo/m.py"]["bare_numbers"] == 1


def test_scan_exempts_tests_config_and_paper(tmp_path):
    for rel in ("tests/t.py", "config/c.py", "paper/p.py", "paper_trade/q.py", "frontend-v2/x.py"):
        path = tmp_path / "trader_koo" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 0.35\n")
    (tmp_path / "trader_koo" / "ok.py").write_text("x = 0.35\n")
    counts = scan(tmp_path)
    assert counts["trader_koo/ok.py"]["bare_numbers"] == 1
    assert counts["trader_koo/config/c.py"]["bare_numbers"] == 0
    assert counts["trader_koo/paper/p.py"]["bare_numbers"] == 0
    assert counts["trader_koo/paper_trade/q.py"]["bare_numbers"] == 0
    assert not any("tests" in k or "frontend-v2" in k for k in counts)


def test_baseline_update_never_raises_numbers():
    base = {"g.py": {"lines": 10, "bare_numbers": 3}}
    new, raised = update_baseline({"g.py": {"lines": 12, "bare_numbers": 1}}, base)
    assert new == {"g.py": {"lines": 10, "bare_numbers": 1}}
    assert raised == ["g.py: lines 12 > baseline 10"]


def test_baseline_update_drops_entries_for_files_that_no_longer_exist():
    new, raised = update_baseline({}, {"gone.py": {"lines": 500, "bare_numbers": 2}})
    assert new == {} and raised == []


def test_seed_records_only_files_that_violate():
    current = {
        "big.py": {"lines": 500, "bare_numbers": 0},
        "num.py": {"lines": 10, "bare_numbers": 4},
        "clean.py": {"lines": 10, "bare_numbers": 0},
    }
    assert set(seed_baseline(current)) == {"big.py", "num.py"}


def test_module_level_named_constants_are_not_bare(tmp_path):
    (tmp_path / "trader_koo").mkdir()
    (tmp_path / "trader_koo" / "m.py").write_text(
        "MAX_DAILY_MOVE = 0.40\n"
        "SPECIAL_DIVIDEND_RATIO: float = 0.10\n"
        "WINDOWS = (21, 63)\n"
        "lower_case = 0.35\n"
        "def f(x):\n    return x > 0.35\n"
    )
    assert scan(tmp_path)["trader_koo/m.py"]["bare_numbers"] == 2
