"""ops/mutation_gate.py: what counts as an equivalent mutant and how a file is scored."""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("mutation_gate", Path(__file__).parents[1] / "ops" / "mutation_gate.py")
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["mutation_gate"] = gate  # dataclasses resolve their module by name
_SPEC.loader.exec_module(gate)

RULES = {"sql_case": True, "log_calls": True, "help_text": True, "equivalent": [
    {"file": "pkg/a.py", "function": "x_f", "original": "x = 1", "mutated": "x = 2", "reason": "fixture"},
]}


def _mutant(original: str, mutated: str, *, line: int = 10, name: str = "pkg.a.x_f__mutmut_1"):
    return gate.Mutant(name, "survived", line, original, mutated)


@pytest.mark.parametrize(
    ("original", "mutated", "equivalent"),
    [
        ('conn.execute("SELECT a FROM t")', 'conn.execute("select a from t")', True),
        ("\"SELECT a FROM t WHERE k = 'daily'\"", "\"SELECT A FROM T WHERE K = 'DAILY'\"", False),
        ('label = "Daily"', 'label = "daily"', False),
        ('p.add_argument("--n", type=int, help="Count")', 'p.add_argument("--n", type=int, help=None)', True),
        ('p.add_argument("--n", type=int, help="Count")', 'p.add_argument("--n", type=int, )', True),
        ('p.add_argument("--n", type=int, help="Count")', 'p.add_argument("--n", type=str, help="Count")', False),
        ("x = 1", "x = 2", True),
        ("x = 1", "x = 3", False),
    ],
)
def test_equivalence_rules_are_narrow(original: str, mutated: str, equivalent: bool) -> None:
    reason = gate.equivalence("pkg/a.py", _mutant(original, mutated), set(), RULES)

    assert (reason is not None) is equivalent


def test_explicit_equivalent_applies_only_to_its_file() -> None:
    assert gate.equivalence("pkg/b.py", _mutant("x = 1", "x = 2"), set(), RULES) is None


def test_mutation_inside_a_logging_call_is_equivalent() -> None:
    mutant = _mutant("int(n or 0),", "int(n or 1),", line=7)

    assert gate.equivalence("pkg/a.py", mutant, {6, 7, 8}, RULES) is not None
    assert gate.equivalence("pkg/a.py", mutant, {6, 8}, RULES) is None


def test_log_call_lines_cover_multi_line_calls_only() -> None:
    tree = gate.ast.parse("LOG.info(\n    'x %d',\n    n,\n)\nprint(n)\nlogger.debug('y')\n")

    assert gate._log_call_lines(tree) == {1, 2, 3, 4, 6}


def test_score_counts_survivors_and_untested_but_not_equivalents() -> None:
    result = gate.FileResult("pkg/a.py", Counter(killed=7, timeout=1, survived=1, **{"no tests": 1, "equivalent": 5, "skipped": 2}))

    assert result.score == 80.0
    assert gate.FileResult("pkg/a.py", Counter(equivalent=3)).score is None


def test_report_fails_below_the_threshold(capsys) -> None:
    low = gate.FileResult("pkg/a.py", Counter(killed=7, survived=3))

    assert gate.report([low], 80.0, verbose=False) is False
    assert gate.report([low], 70.0, verbose=False) is True
    assert "FAIL: pkg/a.py score 70.0%" in capsys.readouterr().out
