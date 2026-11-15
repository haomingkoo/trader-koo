"""ops/mutation_gate.py: what counts as an equivalent mutant, what gets mutated, how a file is scored."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
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


def _src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def _equivalent(source: str, line: int, mutated: str, *, path: str = "pkg/a.py", name: str = "pkg.a.x_f__mutmut_1"):
    """The reason the gate gives for treating *mutated* (replacing *line* of *source*) as equivalent."""
    facts = gate.SourceFacts.parse(source)
    original = facts.lines[line - 1].strip()
    return gate.equivalence(path, gate.Mutant(name, "survived", line, original, mutated), facts, RULES)


# --- sql_case -----------------------------------------------------------------

SQL_SOURCE = _src('''
    def f(conn, k):
        conn.execute("SELECT a FROM t WHERE k = 'daily'")
        raise ValueError("refusing to update: data from the feed is stale")
''')


def test_case_change_inside_a_sql_statement_is_equivalent() -> None:
    assert _equivalent(SQL_SOURCE, 2, '''conn.execute("select a from t where k = 'daily'")''') is not None


def test_case_change_to_a_quoted_sql_value_is_not_equivalent() -> None:
    assert _equivalent(SQL_SOURCE, 2, '''conn.execute("SELECT A FROM T WHERE K = 'DAILY'")''') is None


def test_message_containing_sql_words_is_not_equivalent() -> None:
    assert _equivalent(SQL_SOURCE, 3, 'raise ValueError("REFUSING TO UPDATE: DATA FROM THE FEED IS STALE")') is None


def test_case_change_outside_the_sql_string_is_not_equivalent() -> None:
    source = _src('''
        def f(conn):
            Conn.execute("SELECT a FROM t")
    ''')

    assert _equivalent(source, 2, 'conn.execute("SELECT a FROM t")') is None


def test_sql_split_across_implicitly_concatenated_literals_parses_as_one_statement() -> None:
    source = _src('''
        def f(conn):
            conn.execute(
                "SELECT ticker FROM e WHERE timeframe = 'daily' "
                "AND status IN ('ok', 'skipped')"
            )
    ''')

    assert _equivalent(source, 4, '''"and status in ('ok', 'skipped')"''') is not None


@pytest.mark.parametrize("text", ["SELECT a FROM t WHERE k = ?", "DELETE FROM t", "UPDATE t SET a = 1",
                                  "BEGIN IMMEDIATE", "CREATE TABLE t (a TEXT)"])
def test_sql_statements_parse(text: str) -> None:
    assert gate.parses_as_sql(text)


@pytest.mark.parametrize("text", ["", "update", "data from the feed", "Update from Railway failed",
                                  "SELECT * FROM {table}", "delete the stale rows"])
def test_prose_does_not_parse_as_sql(text: str) -> None:
    assert not gate.parses_as_sql(text)


# --- help_text ----------------------------------------------------------------

ARGPARSE_SOURCE = _src('''
    def main():
        p = argparse.ArgumentParser(description="Run YOLO")
        p.add_argument("--n", type=int, help="Count")
        alert = Alert(description="Price feed is stale", help="Re-run ingest")
''')


@pytest.mark.parametrize("mutated", [
    'p.add_argument("--n", type=int, help=None)',
    'p.add_argument("--n", type=int, )',
    'p.add_argument("--n", type=int, help="XXCountXX")',
])
def test_argparse_help_text_edits_are_equivalent(mutated: str) -> None:
    assert _equivalent(ARGPARSE_SOURCE, 3, mutated) is not None


def test_argument_parser_description_edit_is_equivalent() -> None:
    assert _equivalent(ARGPARSE_SOURCE, 2, 'p = argparse.ArgumentParser(description=None)') is not None


def test_behaviour_edit_on_an_add_argument_line_is_not_equivalent() -> None:
    assert _equivalent(ARGPARSE_SOURCE, 3, 'p.add_argument("--n", type=str, help="Count")') is None


@pytest.mark.parametrize("mutated", [
    'alert = Alert(description=None, help="Re-run ingest")',
    'alert = Alert(description="XXPrice feed is staleXX", help="Re-run ingest")',
    'alert = Alert(description="Price feed is stale", help=None)',
])
def test_description_or_help_on_other_calls_is_not_equivalent(mutated: str) -> None:
    assert _equivalent(ARGPARSE_SOURCE, 4, mutated) is None


# --- log_calls and the explicit list ------------------------------------------

LOG_SOURCE = _src('''
    def f(n):
        LOG.info(
            "x %d",
            int(n or 0),
        )
        return int(n or 0)
''')


def test_mutation_inside_a_logging_call_is_equivalent() -> None:
    assert _equivalent(LOG_SOURCE, 4, "int(n or 1),") is not None


def test_same_mutation_outside_a_logging_call_is_not_equivalent() -> None:
    assert _equivalent(LOG_SOURCE, 6, "return int(n or 1)") is None


def test_log_call_lines_cover_multi_line_calls_only() -> None:
    facts = gate.SourceFacts.parse("LOG.info(\n    'x %d',\n    n,\n)\nprint(n)\nlogger.debug('y')\n")

    assert facts.log_lines == {1, 2, 3, 4, 6}


def test_explicit_equivalent_matches_file_function_and_both_lines() -> None:
    source = "def f():\n    x = 1\n"

    assert _equivalent(source, 2, "x = 2") == "fixture"
    assert _equivalent(source, 2, "x = 3") is None
    assert _equivalent(source, 2, "x = 2", path="pkg/b.py") is None


def test_rules_ignore_a_mutant_whose_line_does_not_match_the_source() -> None:
    facts = gate.SourceFacts.parse(LOG_SOURCE)
    mutant = gate.Mutant("pkg.a.x_f__mutmut_1", "survived", 6, "int(n or 0),", "int(n or 1),")

    assert gate.equivalence("pkg/a.py", mutant, facts, RULES) is None


# --- scoring and the report ---------------------------------------------------

def test_score_counts_survivors_and_untested_but_not_equivalents() -> None:
    result = gate.FileResult("pkg/a.py", Counter(killed=7, timeout=1, survived=1, **{"no tests": 1, "equivalent": 5, "skipped": 2}))

    assert result.score == 80.0
    assert gate.FileResult("pkg/a.py", Counter(equivalent=3)).score is None


def test_report_fails_below_the_threshold(capsys) -> None:
    low = gate.FileResult("pkg/a.py", Counter(killed=7, survived=3))

    assert gate.report([low], 80.0, verbose=False) is False
    assert gate.report([low], 70.0, verbose=False) is True
    assert "FAIL: pkg/a.py score 70.0%" in capsys.readouterr().out


def test_report_fails_a_file_without_mutants_or_a_reason(capsys) -> None:
    assert gate.report([gate.FileResult("pkg/a.py")], 80.0, verbose=False) is False
    assert "FAIL: pkg/a.py produced no mutants" in capsys.readouterr().out


def test_report_passes_a_file_without_mutants_when_it_has_a_reason(capsys) -> None:
    result = gate.FileResult("pkg/a.py", no_mutants_reason="no functions (constants only)")

    assert gate.report([result], 80.0, verbose=False) is True
    assert "pass: pkg/a.py no mutants: no functions (constants only)" in capsys.readouterr().out


# --- collect: missing results fail closed -------------------------------------

@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_collect_fails_a_file_with_functions_but_no_mutmut_results(repo) -> None:
    _write(repo / "pkg/a.py", "def f(x):\n    return x + 1\n")

    (result,) = gate.collect([gate.Target("pkg/a.py")], RULES)

    assert result.score is None and result.no_mutants_reason is None


def test_collect_explains_a_constants_only_file(repo) -> None:
    _write(repo / "pkg/a.py", "LIMIT = 3\nNAMES = ('a', 'b')\n")

    (result,) = gate.collect([gate.Target("pkg/a.py")], RULES)

    assert result.no_mutants_reason == "no functions to mutate (constants and imports only)"


def test_collect_explains_functions_mutmut_never_mutates(repo) -> None:
    _write(repo / "pkg/a.py", "@app.get('/x')\ndef route():\n    return 1\n")

    (result,) = gate.collect([gate.Target("pkg/a.py")], RULES)

    assert result.no_mutants_reason == "mutmut skips decorated functions: x_route"


def test_collect_fails_scoped_functions_that_produced_no_mutants(repo) -> None:
    _write(repo / "pkg/a.py", "def f(x):\n    return x + 1\n\ndef g(x):\n    return x - 1\n")
    _write(repo / "mutants/pkg/a.py.meta", json.dumps({"exit_code_by_key": {"pkg.a.x_f__mutmut_1": 1}}))

    (result,) = gate.collect([gate.Target("pkg/a.py", frozenset({"x_g"}))], RULES)

    assert result.score is None and result.no_mutants_reason is None


def test_collect_scores_only_the_scoped_functions(repo) -> None:
    source = "def f(x):\n    return x + 1\n\ndef g(x):\n    return x - 1\n"
    _write(repo / "pkg/a.py", source)
    _write(repo / "mutants/pkg/a.py", source)  # describe() finds no variants; status alone is scored
    codes = {"pkg.a.x_f__mutmut_1": 0, "pkg.a.x_g__mutmut_1": 1, "pkg.a.x_g__mutmut_2": 1}
    _write(repo / "mutants/pkg/a.py.meta", json.dumps({"exit_code_by_key": codes}))

    (scoped,) = gate.collect([gate.Target("pkg/a.py", frozenset({"x_g"}))], RULES)
    (whole,) = gate.collect([gate.Target("pkg/a.py")], RULES)

    assert scoped.score == 100.0 and dict(scoped.counts) == {"killed": 2}
    assert whole.counts["survived"] == 1


# --- diff parsing and function scoping ----------------------------------------

DIFF = _src('''
    diff --git a/pkg/a.py b/pkg/a.py
    index 1..2 100644
    --- a/pkg/a.py
    +++ b/pkg/a.py
    @@ -3,0 +4,2 @@ def f():
    +    y = 2
    +    z = 3
    @@ -10,2 +11,0 @@ def g():
    -    a = 1
    -    b = 2
    @@ -20 +19 @@ def h():
    -    c = 1
    +    c = 2
    diff --git a/tests/test_old.py b/tests/test_old.py
    deleted file mode 100644
    --- a/tests/test_old.py
    +++ /dev/null
    @@ -1,2 +0,0 @@
    -import x
    -x.f()
    diff --git a/tests/test_a.py b/tests/test_b.py
    similarity index 90%
    rename from tests/test_a.py
    rename to tests/test_b.py
    --- a/tests/test_a.py
    +++ b/tests/test_b.py
    @@ -5 +5 @@
    -    assert f() == 1
    +    assert f()
''')


def test_parse_diff_maps_hunks_to_old_and_new_lines() -> None:
    files = {d.new_path or d.old_path: d for d in gate.parse_diff(DIFF)}

    a = files["pkg/a.py"]
    assert a.new_lines == {4, 5, 11, 12, 19}  # a pure deletion marks the lines around it
    assert a.old_lines == {3, 4, 10, 11, 20}
    old = files["tests/test_old.py"]
    assert (old.old_path, old.new_path, old.old_lines) == ("tests/test_old.py", None, {1, 2})
    renamed = files["tests/test_b.py"]
    assert (renamed.old_path, renamed.old_lines, renamed.new_lines) == ("tests/test_a.py", {5}, {5})


UNITS_SOURCE = _src('''
    LIMIT = 3


    def f():
        return 1


    @staticmethod
    def g():
        return 2


    class C:
        @classmethod
        def m(cls):
            return 3

        @property
        def p(self):
            return 4
''')


def test_units_name_functions_the_way_mutmut_does() -> None:
    units = {u.key: u for u in gate.units(UNITS_SOURCE)}

    assert set(units) == {"x_f", "x_g", "xǁCǁm", "xǁCǁp"}
    assert (units["x_f"].start, units["x_f"].end) == (4, 5)
    assert (units["x_g"].start, units["x_g"].end) == (8, 10)  # decorators belong to the function
    assert units["xǁCǁm"].mutable and units["x_g"].mutable
    assert not units["xǁCǁp"].mutable


def test_scope_maps_changed_lines_to_their_enclosing_functions() -> None:
    assert gate.functions_touching(UNITS_SOURCE, {5, 15}) == {"x_f", "xǁCǁm"}
    assert gate.functions_touching(UNITS_SOURCE, {1, 2}) == set()


def test_small_files_are_mutated_whole_and_large_files_by_function(repo) -> None:
    _write(repo / "pkg/small.py", UNITS_SOURCE)
    _write(repo / "pkg/large.py", UNITS_SOURCE + "\n" * gate.WHOLE_FILE_MAX_LINES)

    assert gate.source_target("pkg/small.py", {5}) == gate.Target("pkg/small.py")
    assert gate.source_target("pkg/large.py", {5}) == gate.Target("pkg/large.py", frozenset({"x_f"}))
    assert gate.source_target("pkg/large.py", {1}) == gate.Target("pkg/large.py", frozenset())


def test_mutant_globs_follow_the_scope() -> None:
    assert gate.mutant_globs(gate.Target("trader_koo/a/b.py")) == ["trader_koo.a.b.*"]
    assert gate.mutant_globs(gate.Target("trader_koo/a/b.py", frozenset({"x_f", "xǁCǁm"}))) == [
        "trader_koo.a.b.x_f__mutmut_*", "trader_koo.a.b.xǁCǁm__mutmut_*",
    ]


def test_merging_targets_keeps_the_widest_scope() -> None:
    merged = gate.merge_targets([
        gate.Target("pkg/a.py", frozenset({"x_f"})),
        gate.Target("pkg/a.py", frozenset({"x_g"})),
        gate.Target("pkg/b.py", frozenset({"x_f"})),
        gate.Target("pkg/b.py"),
    ])

    assert merged == [gate.Target("pkg/a.py", frozenset({"x_f", "x_g"})), gate.Target("pkg/b.py")]


# --- tests a PR deletes or edits ----------------------------------------------

TEST_SOURCE = _src('''
    from unittest.mock import patch

    from trader_koo.report import runs
    from trader_koo.report.freshness import assess_data_freshness as assess


    def _helper():
        return runs.start_report_run()


    def test_complete():
        assert runs.complete_report_run() == 1


    def test_assess():
        assert assess() is not None


    class TestFail:
        def test_fail(self):
            with patch("trader_koo.report.runs.fail_report_run"):
                _helper()
''')


@pytest.fixture
def sources(repo):
    _write(repo / "trader_koo/report/runs.py", _src('''
        def start_report_run():
            return 1

        def complete_report_run():
            return 1

        def fail_report_run():
            return 1

        class Runner:
            def go(self):
                return 1
    '''))
    _write(repo / "trader_koo/report/freshness.py", "def assess_data_freshness():\n    return 1\n")
    _write(repo / "trader_koo/report/__init__.py", "")
    return repo


def test_edited_test_function_targets_the_functions_it_references(sources) -> None:
    refs = gate.referenced_functions(TEST_SOURCE, {12})

    assert refs == {"trader_koo/report/runs.py": {"complete_report_run"}}


def test_aliased_imports_and_patch_targets_resolve(sources) -> None:
    assert gate.referenced_functions(TEST_SOURCE, {16}) == {"trader_koo/report/freshness.py": {"assess_data_freshness"}}
    assert gate.referenced_functions(TEST_SOURCE, {21}) == {"trader_koo/report/runs.py": {"fail_report_run"}}


def test_edited_import_targets_only_what_it_imports(sources) -> None:
    assert gate.referenced_functions(TEST_SOURCE, {4}) == {"trader_koo/report/freshness.py": {"assess_data_freshness"}}


def test_edited_module_import_targets_everything_used_through_it(sources) -> None:
    assert gate.referenced_functions(TEST_SOURCE, {3}) == {
        "trader_koo/report/runs.py": {"start_report_run", "complete_report_run", "fail_report_run"},
    }


def test_blank_or_comment_lines_between_tests_select_nothing(sources) -> None:
    source = TEST_SOURCE.replace("\n\ndef test_assess", "\n# assess\ndef test_assess")

    assert gate.referenced_functions(source, {13, 14}) == {}


def test_edit_outside_a_test_function_targets_everything_the_file_references(sources) -> None:
    source = TEST_SOURCE.replace("from unittest.mock import patch", "LIMIT = 3\nfrom unittest.mock import patch")
    refs = gate.referenced_functions(source, {1})

    assert refs == {
        "trader_koo/report/runs.py": {"start_report_run", "complete_report_run", "fail_report_run"},
        "trader_koo/report/freshness.py": {"assess_data_freshness"},
    }


def test_test_function_reaching_code_only_through_helpers_falls_back_to_the_whole_file(sources) -> None:
    source = TEST_SOURCE + "\n\ndef test_indirect():\n    assert _helper()\n"
    line = len(source.splitlines())

    assert gate.referenced_functions(source, {line})["trader_koo/report/runs.py"] >= {"start_report_run"}


def test_targets_from_tests_scope_large_modules_and_take_small_ones_whole(sources) -> None:
    _write(sources / "trader_koo/report/runs.py",
           (sources / "trader_koo/report/runs.py").read_text() + "\n" * gate.WHOLE_FILE_MAX_LINES)

    targets = gate.targets_from_tests(TEST_SOURCE, {3, 4})

    assert gate.Target("trader_koo/report/runs.py",
                       frozenset({"x_start_report_run", "x_complete_report_run", "x_fail_report_run"})) in targets
    assert gate.Target("trader_koo/report/freshness.py") in targets


def _git(*args: str) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True)


def test_deleted_and_edited_tests_select_the_modules_they_imported(sources) -> None:
    _git("init", "-q", "-b", "main")
    _git("config", "user.email", "gate@test")
    _git("config", "user.name", "gate")
    _write(sources / "tests/test_runs.py", TEST_SOURCE)
    _write(sources / "tests/test_gone.py", "from trader_koo.report import runs\n\n\ndef test_go():\n    runs.Runner().go()\n")
    _git("add", "-A")
    _git("commit", "-qm", "base")
    _git("checkout", "-qb", "feature")
    (sources / "tests/test_gone.py").unlink()
    edited = TEST_SOURCE.replace("== 1", "")
    _write(sources / "tests/test_runs.py", edited)
    _write(sources / "tests/test_new.py", "from trader_koo.report import freshness\n")
    _git("add", "-A")
    _git("commit", "-qm", "weaken tests")

    targets = gate.changed_targets("main")

    assert targets == [
        gate.Target("trader_koo/report/runs.py"),  # small file: whole
    ]
    assert gate.changed_targets("main", whole_file_max_lines=0) == [
        gate.Target("trader_koo/report/runs.py", frozenset({"x_complete_report_run", "xǁRunnerǁgo"})),
    ]
