"""Mutation-test the Python modules a PR changed and fail below a score.

Usage:
  python ops/mutation_gate.py --base origin/main [--threshold 80]
  python ops/mutation_gate.py --paths trader_koo/report/freshness.py [--only 'glob' ...]

Picks changed ``trader_koo/**/*.py`` files (added or modified versus the merge
base), points ``[tool.mutmut] only_mutate`` at them and the test selection at the
test files that import them, runs mutmut and scores each file as
(killed + timeout) / (mutants - skipped - equivalent). Survivors and mutants no
test reaches count against the score. A changed module that no test imports
fails the gate.

Equivalent mutants cannot change behaviour, so they leave the denominator. They
are recognised only by the rules in ``[tool.mutation_gate]`` of pyproject.toml:
three built-in rules (case-only edits to SQL text, edits inside logging calls,
edits to argparse help or description text) and an explicit list matched on
the file, function, original and mutated line text, each entry with its reason.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_ROOT = "trader_koo/"
EXCLUDED_PREFIXES = ("trader_koo/frontend-v2/",)
PYPROJECT = Path("pyproject.toml")
MUTANTS = Path("mutants")
STATUS = {1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests", 34: "skipped",
          36: "timeout", 24: "timeout", -24: "timeout", 152: "timeout", 255: "timeout", None: "not checked"}
LOG_NAMES = {"LOG", "logger", "log", "LOGGER"}
LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}
SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|FROM|WHERE|BEGIN|COMMIT|ROLLBACK)\b", re.IGNORECASE)
DOC_ARG_RE = re.compile(r"""\b(?:help|description)=(?:None|(["']).*?\1)""")


def _list_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^{key} = \[.*?^\]", re.MULTILINE | re.DOTALL)


def changed_sources(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", f"{base}...HEAD", "--", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(
        path for path in out.split()
        if path.startswith(SOURCE_ROOT) and not path.startswith(EXCLUDED_PREFIXES) and Path(path).is_file()
    )


def importing_tests(source: str) -> list[str]:
    """Test files that import *source* by dotted path or ``from pkg import mod``."""
    dotted = source.removesuffix(".py").removesuffix("/__init__").replace("/", ".")
    package, _, module = dotted.rpartition(".")
    pattern = re.compile(
        rf"{re.escape(dotted)}(?![\w])"
        rf"|from {re.escape(package)} import (?:[^#\n]*[ ,(])?{re.escape(module)}(?![\w])"
    )
    return sorted(
        str(path) for path in Path("tests").rglob("test_*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )


def set_mutmut_list(text: str, key: str, values: list[str]) -> str:
    listing = f"{key} = [\n" + "".join(f'    "{v}",\n' for v in values) + "]"
    updated, count = _list_re(key).subn(listing, text, count=1)
    if count != 1:
        sys.exit(f"mutation gate: no {key} list under [tool.mutmut] in pyproject.toml")
    return updated


@dataclass
class Mutant:
    name: str
    status: str
    line: int | None = None
    original: str = ""
    mutated: str = ""


@dataclass
class FileResult:
    path: str
    counts: Counter = field(default_factory=Counter)
    survivors: list[Mutant] = field(default_factory=list)
    equivalent: list[tuple[Mutant, str]] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        tested = sum(self.counts.values()) - self.counts["skipped"] - self.counts["equivalent"]
        return None if tested <= 0 else 100.0 * (self.counts["killed"] + self.counts["timeout"]) / tested


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _log_call_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        func = getattr(node, "func", None) if isinstance(node, ast.Call) else None
        if (isinstance(func, ast.Attribute) and func.attr in LOG_LEVELS
                and isinstance(func.value, ast.Name) and func.value.id in LOG_NAMES):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def _original_name(mangled: str) -> str:
    base = mangled.split("__mutmut_")[0]
    return base.rsplit("ǁ", 1)[-1] if "ǁ" in base else base.removeprefix("x_")


def describe(path: str, keys: list[str]) -> dict[str, tuple[int | None, str, str]]:
    """Map each mutant name to (original line number, original text, mutated text)."""
    mutant_src = (MUTANTS / path).read_text(encoding="utf-8")
    mutant_lines = mutant_src.splitlines()
    mutant_funcs = _functions(ast.parse(mutant_src))
    original_funcs = _functions(ast.parse(Path(path).read_text(encoding="utf-8")))
    out: dict[str, tuple[int | None, str, str]] = {}
    for key in keys:
        mangled = key.rsplit(".", 1)[1]
        orig_copy = mutant_funcs.get(mangled.split("__mutmut_")[0] + "__mutmut_orig")
        variant = mutant_funcs.get(mangled)
        source_def = original_funcs.get(_original_name(mangled))
        if orig_copy is None or variant is None:
            out[key] = (None, "", "")
            continue
        before = mutant_lines[orig_copy.lineno:orig_copy.end_lineno]
        after = mutant_lines[variant.lineno:variant.end_lineno]
        ops = [op for op in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes() if op[0] != "equal"]
        if not ops:
            out[key] = (None, "", "")
            continue
        _, i1, i2, j1, j2 = ops[0]
        line = source_def.lineno + 1 + i1 if source_def is not None else None
        out[key] = (line, " | ".join(s.strip() for s in before[i1:i2]),
                    " | ".join(s.strip() for s in after[j1:j2]))
    return out


def _without_doc_args(text: str) -> str:
    return re.sub(r"[\s,|]", "", DOC_ARG_RE.sub("", text))


def equivalence(path: str, mutant: Mutant, log_lines: set[int], rules: dict) -> str | None:
    original, mutated = mutant.original, mutant.mutated
    if original == mutated:
        return None
    if (rules.get("sql_case") and original.lower() == mutated.lower() and SQL_RE.search(original)
            and re.findall(r"'[^']*'", original) == re.findall(r"'[^']*'", mutated)):
        return "SQL keywords and identifiers are case-insensitive in SQLite"
    if rules.get("log_calls") and mutant.line is not None and mutant.line in log_lines:
        return "only changes a logging call's arguments"
    if rules.get("help_text") and _without_doc_args(original) == _without_doc_args(mutated):
        return "only changes argparse help or description text"
    for entry in rules.get("equivalent", []):
        if (entry["file"] == path and fnmatch.fnmatch(mutant.name, f"*.{entry['function']}__mutmut_*")
                and entry["original"] == original and entry["mutated"] == mutated):
            return entry["reason"]
    return None


def collect(paths: list[str], only: list[str], rules: dict) -> list[FileResult]:
    results = []
    for path in paths:
        result = FileResult(path)
        meta = MUTANTS / f"{path}.meta"
        if not meta.exists():
            results.append(result)
            continue
        codes = json.loads(meta.read_text(encoding="utf-8"))["exit_code_by_key"]
        keys = [k for k in codes if not only or any(fnmatch.fnmatch(k, pat) for pat in only)]
        details = describe(path, keys)
        log_lines = _log_call_lines(ast.parse(Path(path).read_text(encoding="utf-8")))
        for key in sorted(keys, key=lambda k: (k.split("__mutmut_")[0], int(k.rsplit("_", 1)[1]))):
            line, original, mutated = details[key]
            mutant = Mutant(key, STATUS.get(codes[key], "suspicious"), line, original, mutated)
            if mutant.status in ("survived", "no tests"):
                reason = equivalence(path, mutant, log_lines, rules)
                if reason:
                    result.counts["equivalent"] += 1
                    result.equivalent.append((mutant, reason))
                    continue
                result.survivors.append(mutant)
            result.counts[mutant.status] += 1
        results.append(result)
    return results


def report(results: list[FileResult], threshold: float, *, verbose: bool) -> bool:
    passed = True
    for result in results:
        score = result.score
        verdict = "no mutants" if score is None else ("pass" if score >= threshold else "FAIL")
        passed &= score is None or score >= threshold
        shown = "n/a" if score is None else f"{score:.1f}%"
        print(f"mutation gate: {verdict}: {result.path} score {shown} {dict(result.counts)}")
        for mutant in result.survivors:
            print(f"  {mutant.status}: {result.path}:{mutant.line} {mutant.name.rsplit('.', 1)[1]}: "
                  f"{mutant.original[:120]}  ==>  {mutant.mutated[:120]}")
        if verbose:
            for mutant, reason in result.equivalent:
                print(f"  equivalent ({reason}): {result.path}:{mutant.line}: {mutant.mutated[:120]}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="score Python sources changed versus this ref")
    source.add_argument("--paths", nargs="+", help="score these sources instead of a diff")
    parser.add_argument("--only", nargs="*", default=[], help="mutant name globs to run and score")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--no-run", action="store_true", help="score the existing mutants/ results")
    parser.add_argument("--verbose", action="store_true", help="also list equivalent mutants")
    args = parser.parse_args()

    paths = args.paths or changed_sources(args.base)
    if not paths:
        print(f"mutation gate: skipped: no Python source under {SOURCE_ROOT} changed versus {args.base}")
        return 0
    rules = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("tool", {}).get("mutation_gate", {})
    if not args.no_run:
        tests_by_source = {path: importing_tests(path) for path in paths}
        untested = [path for path, tests in tests_by_source.items() if not tests]
        if untested:
            print("mutation gate: FAIL: no test file imports", *untested, sep="\n  ")
            return 1
        tests = sorted({t for found in tests_by_source.values() for t in found})
        print("mutation gate: mutating", *paths, sep="\n  ")
        print("mutation gate: with tests", *tests, sep="\n  ")
        text = PYPROJECT.read_text(encoding="utf-8")
        text = set_mutmut_list(text, "only_mutate", paths)
        text = set_mutmut_list(text, "pytest_add_cli_args_test_selection", tests)
        PYPROJECT.write_text(text, encoding="utf-8")
        run = subprocess.run(["mutmut", "run", *args.only], check=False)
        if run.returncode != 0:
            print(f"mutation gate: FAIL: mutmut run exited {run.returncode}")
            return run.returncode
    ok = report(collect(paths, args.only, rules), args.threshold, verbose=args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
