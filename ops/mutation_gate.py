"""Mutation-test the Python code a PR changed and fail below a score.

Usage:
  python ops/mutation_gate.py --base origin/main [--threshold 80]
  python ops/mutation_gate.py --paths trader_koo/report/freshness.py [--functions x_name ...]

What gets mutated (``--base``):
- ``trader_koo/**/*.py`` files the PR adds, modifies or renames. A file of at
  most ``WHOLE_FILE_MAX_LINES`` lines is mutated whole; a larger one only in
  the functions whose lines the diff touched.
- The ``trader_koo/`` functions referenced by test code the PR deletes or
  edits, so weakening or deleting a test cannot skip the gate. An edit inside
  a test function selects what that function references; an edit outside any
  function (imports, module helpers) or a test that reaches code only through
  helpers selects everything the test file references.

Each file is scored as (killed + timeout) / (mutants - skipped - equivalent)
over the scoped functions. Survivors and mutants no test reaches count
against the score. A mutated module that no test imports fails, and so does a
scoped file that produced no mutmut results, unless it has a reason to have
none: constants only, changed lines outside any function, or functions mutmut
never mutates (decorated other than ``@staticmethod``/``@classmethod``).

Equivalent mutants cannot change behaviour, so they leave the denominator. They
are recognised only by the rules in ``[tool.mutation_gate]`` of pyproject.toml:
three built-in rules (case-only edits inside string literals that parse as a
SQL statement, edits inside logging calls, edits to the help or description
text of argparse ``add_argument``/``ArgumentParser`` calls) and an explicit
list matched on the file, function, original and mutated line text, each entry
with its reason.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import json
import re
import sqlite3
import subprocess
import sys
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_ROOT = "trader_koo/"
TEST_ROOT = "tests/"
EXCLUDED_PREFIXES = ("trader_koo/frontend-v2/",)
# Files up to this size are mutated whole; larger ones only in touched functions.
WHOLE_FILE_MAX_LINES = 300
PYPROJECT = Path("pyproject.toml")
MUTANTS = Path("mutants")
STATUS = {1: "killed", 3: "killed", 0: "survived", 5: "no tests", 33: "no tests", 34: "skipped",
          36: "timeout", 24: "timeout", -24: "timeout", 152: "timeout", 255: "timeout", None: "not checked"}
LOG_NAMES = {"LOG", "logger", "log", "LOGGER"}
LOG_LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}
ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
SEP = "ǁ"  # mutmut's class separator in mangled names: xǁClassǁmethod
DOC_ARG_RE = re.compile(r"""\b(?:help|description)=(?:None|(["']).*?\1)""")
SQL_NOT_PARSED = ("syntax error", "incomplete input", "unrecognized token")
DOTTED_SOURCE_RE = re.compile(r"trader_koo(?:\.\w+)+")


# --- what to mutate -----------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """A source file to mutate: whole when *functions* is None, else these mangled names."""
    path: str
    functions: frozenset[str] | None = None


@dataclass(frozen=True)
class Unit:
    """A function mutmut can name: ``x_f`` or ``xǁClassǁmethod``, with its line span."""
    key: str
    start: int
    end: int
    mutable: bool
    def_line: int = 0


@dataclass
class FileDiff:
    old_path: str | None
    new_path: str | None
    old_lines: set[int] = field(default_factory=set)
    new_lines: set[int] = field(default_factory=set)


def _mutmut_mutates(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """mutmut 3 skips decorated functions except a lone @staticmethod or @classmethod."""
    decorators = node.decorator_list
    return not decorators or (
        len(decorators) == 1 and isinstance(decorators[0], ast.Name)
        and decorators[0].id in ("staticmethod", "classmethod")
    )


def _unit_nodes(tree: ast.Module) -> list[tuple[Unit, ast.AST]]:
    out: list[tuple[Unit, ast.AST]] = []

    def visit(body: list[ast.stmt], cls: str, enum: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = f"x{SEP}{cls}{SEP}{node.name}" if cls else f"x_{node.name}"
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                out.append((Unit(key, start, node.end_lineno, _mutmut_mutates(node) and not enum, node.lineno), node))
            elif isinstance(node, ast.ClassDef):
                bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
                visit(node.body, f"{cls}.{node.name}" if cls else node.name, enum or bool(bases & ENUM_BASES))

    visit(tree.body, "", False)
    return out


def units(source: str) -> list[Unit]:
    return [unit for unit, _ in _unit_nodes(ast.parse(source))]


def functions_touching(source: str, lines: set[int]) -> set[str]:
    return {u.key for u in units(source) if any(u.start <= n <= u.end for n in lines)}


def _line_count(path: str) -> int:
    return len(Path(path).read_text(encoding="utf-8").splitlines())


def source_target(path: str, lines: set[int], whole_file_max_lines: int = WHOLE_FILE_MAX_LINES) -> Target:
    if _line_count(path) <= whole_file_max_lines:
        return Target(path)
    return Target(path, frozenset(functions_touching(Path(path).read_text(encoding="utf-8"), lines)))


def merge_targets(targets: list[Target]) -> list[Target]:
    merged: dict[str, frozenset[str] | None] = {}
    for target in targets:
        if target.path not in merged:
            merged[target.path] = target.functions
        elif merged[target.path] is not None:
            merged[target.path] = None if target.functions is None else merged[target.path] | target.functions
    return [Target(path, functions) for path, functions in sorted(merged.items())]


def module_name(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def mutant_globs(target: Target) -> list[str]:
    module = module_name(target.path)
    if target.functions is None:
        return [f"{module}.*"]
    return [f"{module}.{key}__mutmut_*" for key in sorted(target.functions)]


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _hunk_lines(start: int, count: int) -> set[int]:
    if count:
        return set(range(start, start + count))
    return {max(start, 1), start + 1}  # a pure insertion or deletion touches the lines around it


def parse_diff(text: str) -> list[FileDiff]:
    """Changed line numbers per file from ``git diff -U0`` output, on both sides."""
    files: list[FileDiff] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            match = re.match(r"diff --git a/(.+) b/(.+)$", line)
            files.append(FileDiff(match.group(1), match.group(2)) if match else FileDiff(None, None))
        elif not files:
            continue
        elif line.startswith("--- "):
            files[-1].old_path = None if line == "--- /dev/null" else line[6:]
        elif line.startswith("+++ "):
            files[-1].new_path = None if line == "+++ /dev/null" else line[6:]
        elif match := _HUNK_RE.match(line):
            a, b, c, d = match.groups()
            files[-1].old_lines |= _hunk_lines(int(a), 1 if b is None else int(b))
            files[-1].new_lines |= _hunk_lines(int(c), 1 if d is None else int(d))
    return files


def _is_source(path: str | None) -> bool:
    return bool(path) and path.startswith(SOURCE_ROOT) and path.endswith(".py") and not path.startswith(EXCLUDED_PREFIXES)


def _is_test(path: str | None) -> bool:
    return bool(path) and path.startswith(TEST_ROOT) and path.endswith(".py")


def _module_path(parts: list[str]) -> str | None:
    for candidate in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if _is_source(candidate) and Path(candidate).is_file():
            return candidate
    return None


_DEFS: dict[str, set[str]] = {}


def _top_level_defs(path: str) -> set[str]:
    if path not in _DEFS:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        _DEFS[path] = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    return _DEFS[path]


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


def _resolve(dotted: str) -> tuple[str, str] | None:
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        path = _module_path(parts[:i])
        if path:
            return (path, parts[i]) if parts[i] in _top_level_defs(path) else None
    return None


def _references(nodes: list[ast.AST], aliases: dict[str, str],
                modules: set[str] | None = None) -> dict[str, set[str]]:
    """Top-level defs that *nodes* reach through *aliases* or dotted strings (patch targets).

    With *modules*, dotted strings count only when they point into one of those modules.
    """
    found: dict[str, set[str]] = {}
    for root in nodes:
        for node in ast.walk(root):
            from_string = isinstance(node, ast.Constant) and isinstance(node.value, str)
            if isinstance(node, (ast.Name, ast.Attribute)):
                candidates = [_dotted(node, aliases)]
            elif from_string:
                candidates = DOTTED_SOURCE_RE.findall(node.value)
            else:
                continue
            for dotted in filter(None, candidates):
                resolved = _resolve(dotted)
                if resolved and not (from_string and modules is not None and resolved[0] not in modules):
                    found.setdefault(resolved[0], set()).add(resolved[1])
    return found


def _merge_refs(*refs: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for ref in refs:
        for path, names in ref.items():
            merged.setdefault(path, set()).update(names)
    return merged


def _import_aliases(nodes: list[ast.stmt]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("trader_koo"):
                    local = alias.asname or alias.name.split(".")[0]
                    aliases[local] = alias.name if alias.asname else local
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith("trader_koo"):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def referenced_functions(test_source: str, lines: set[int]) -> dict[str, set[str]]:
    """``trader_koo`` modules and their top-level functions or classes the changed test code uses.

    A changed line inside a function selects what that function references (or the
    whole file when it references nothing directly, so it works through helpers).
    A changed import selects what is used through the names it binds. Any other
    module-level change selects everything the file references.
    """
    tree = ast.parse(test_source)
    aliases = _import_aliases(list(ast.walk(tree)))
    spans = _unit_nodes(tree)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    source_lines = test_source.splitlines()
    enclosing, changed_imports, module_level = [], [], False
    for n in sorted(lines):
        inside = [node for unit, node in spans if unit.start <= n <= unit.end]
        hit_imports = [node for node in imports if node.lineno <= n <= node.end_lineno]
        enclosing += [node for node in inside if node not in enclosing]
        changed_imports += [node for node in hit_imports if node not in changed_imports]
        text = source_lines[n - 1].strip() if 0 < n <= len(source_lines) else ""
        module_level |= not inside and not hit_imports and bool(text) and not text.startswith("#")
    if module_level:
        return _references([tree], aliases)
    refs = []
    if enclosing:
        refs.append(_references(enclosing, aliases) or _references([tree], aliases))
    if changed_imports:
        bound = _import_aliases(changed_imports)
        modules = {path for dotted in bound.values() for path in [_module_path(dotted.split("."))] if path}
        modules |= {r[0] for dotted in bound.values() for r in [_resolve(dotted)] if r}
        refs.append(_references([tree], bound, modules))
    return _merge_refs(*refs)


def targets_from_tests(test_source: str, lines: set[int],
                       whole_file_max_lines: int = WHOLE_FILE_MAX_LINES) -> list[Target]:
    targets = []
    for path, names in referenced_functions(test_source, lines).items():
        if _line_count(path) <= whole_file_max_lines:
            targets.append(Target(path))
            continue
        keys = {u.key for u in units(Path(path).read_text(encoding="utf-8"))
                if u.key.removeprefix("x_") in names or u.key.split(SEP)[1:2] and u.key.split(SEP)[1] in names}
        targets.append(Target(path, frozenset(keys)))
    return targets


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def changed_targets(base: str, whole_file_max_lines: int = WHOLE_FILE_MAX_LINES) -> list[Target]:
    merge_base = _git("merge-base", base, "HEAD").strip()
    diff = _git("diff", "-U0", "--no-color", "--no-ext-diff", "--find-renames", merge_base, "HEAD", "--", "*.py")
    targets: list[Target] = []
    for change in parse_diff(diff):
        if _is_source(change.new_path) and Path(change.new_path).is_file() and change.new_lines:
            targets.append(source_target(change.new_path, change.new_lines, whole_file_max_lines))
        if not _is_test(change.old_path):
            continue  # added tests cannot weaken the suite
        if change.old_lines:
            old_source = _git("show", f"{merge_base}:{change.old_path}")
            targets += targets_from_tests(old_source, change.old_lines, whole_file_max_lines)
        if _is_test(change.new_path) and change.new_lines:
            new_source = Path(change.new_path).read_text(encoding="utf-8")
            targets += targets_from_tests(new_source, change.new_lines, whole_file_max_lines)
    return merge_targets(targets)


def importing_tests(source: str) -> list[str]:
    """Test files that import *source* by dotted path or ``from pkg import mod``."""
    dotted = module_name(source)
    package, _, module = dotted.rpartition(".")
    pattern = re.compile(
        rf"{re.escape(dotted)}(?![\w])"
        rf"|from {re.escape(package)} import (?:[^#\n]*[ ,(])?{re.escape(module)}(?![\w])"
    )
    return sorted(
        str(path) for path in Path("tests").rglob("test_*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    )


def _list_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^{key} = \[.*?^\]", re.MULTILINE | re.DOTALL)


def set_mutmut_list(text: str, key: str, values: list[str]) -> str:
    listing = f"{key} = [\n" + "".join(f'    "{v}",\n' for v in values) + "]"
    updated, count = _list_re(key).subn(listing, text, count=1)
    if count != 1:
        sys.exit(f"mutation gate: no {key} list under [tool.mutmut] in pyproject.toml")
    return updated


# --- equivalence --------------------------------------------------------------

def parses_as_sql(text: str) -> bool:
    """True when SQLite parses *text* as a statement. Nothing runs: the authorizer denies every action."""
    if not text.strip():
        return False
    conn = sqlite3.connect(":memory:")
    conn.set_authorizer(lambda *_: sqlite3.SQLITE_DENY)
    try:
        conn.execute(text)
    except sqlite3.Warning:
        return True  # a second statement follows a parsed first one
    except sqlite3.Error as exc:
        return not any(marker in str(exc) for marker in SQL_NOT_PARSED)
    finally:
        conn.close()
    return True


def _char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


@dataclass
class SourceFacts:
    """What the equivalence rules need to know about the original source."""
    lines: list[str]
    log_lines: set[int]
    argparse_lines: set[int]
    # (start line, start col, end line, end col, value), cols in characters
    strings: list[tuple[int, int, int, int, str]]

    @classmethod
    def parse(cls, text: str) -> SourceFacts:
        tree = ast.parse(text)
        lines = text.splitlines()
        log_lines: set[int] = set()
        argparse_lines: set[int] = set()
        in_fstring = {id(c) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for c in ast.walk(n)}
        strings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                span = range(node.lineno, node.end_lineno + 1)
                if (isinstance(func, ast.Attribute) and func.attr in LOG_LEVELS
                        and isinstance(func.value, ast.Name) and func.value.id in LOG_NAMES):
                    log_lines.update(span)
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in ("add_argument", "ArgumentParser"):
                    argparse_lines.update(span)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in in_fstring:
                strings.append((node.lineno, _char_col(lines[node.lineno - 1], node.col_offset),
                                node.end_lineno, _char_col(lines[node.end_lineno - 1], node.end_col_offset),
                                node.value))
        return cls(lines, log_lines, argparse_lines, strings)

    def string_at(self, line: int, col: int) -> str | None:
        for start_line, start_col, end_line, end_col, value in self.strings:
            if (start_line, start_col) <= (line, col) < (end_line, end_col):
                return value
        return None


@dataclass
class Mutant:
    name: str
    status: str
    line: int | None = None
    original: str = ""
    mutated: str = ""


def _sql_case_only(mutant: Mutant, facts: SourceFacts) -> bool:
    original, mutated = mutant.original, mutant.mutated
    if len(original) != len(mutated) or original.lower() != mutated.lower():
        return False
    if re.findall(r"'[^']*'", original) != re.findall(r"'[^']*'", mutated):
        return False  # quoted SQL values are case-sensitive
    raw = facts.lines[mutant.line - 1]
    indent = len(raw) - len(raw.lstrip())
    for i, (a, b) in enumerate(zip(original, mutated)):
        if a != b:
            sql = facts.string_at(mutant.line, indent + i)
            if sql is None or not parses_as_sql(sql):
                return False
    return True


def _without_doc_args(text: str) -> str:
    return re.sub(r"[\s,|]", "", DOC_ARG_RE.sub("", text))


def equivalence(path: str, mutant: Mutant, facts: SourceFacts, rules: dict) -> str | None:
    original, mutated = mutant.original, mutant.mutated
    if original == mutated:
        return None
    # Built-in rules read the source, so they apply only when the original text is
    # exactly the source lines starting at the mutant's line (describe() joins lines with " | ").
    parts = original.split(" | ")
    span = set(range(mutant.line, mutant.line + len(parts))) if mutant.line else set()
    line_matches = bool(span) and max(span) <= len(facts.lines) and all(
        facts.lines[mutant.line - 1 + k].strip() == part for k, part in enumerate(parts)
    )
    if line_matches:
        if rules.get("sql_case") and len(parts) == 1 and _sql_case_only(mutant, facts):
            return "case-only edit inside a SQL statement; SQLite keywords and identifiers are case-insensitive"
        if rules.get("log_calls") and span <= facts.log_lines:
            return "only changes a logging call's arguments"
        if (rules.get("help_text") and span <= facts.argparse_lines
                and _without_doc_args(original) == _without_doc_args(mutated)):
            return "only changes argparse help or description text"
    for entry in rules.get("equivalent", []):
        if (entry["file"] == path and fnmatch.fnmatch(mutant.name, f"*.{entry['function']}__mutmut_*")
                and entry["original"] == original and entry["mutated"] == mutated):
            return entry["reason"]
    return None


# --- results ------------------------------------------------------------------

@dataclass
class FileResult:
    path: str
    counts: Counter = field(default_factory=Counter)
    survivors: list[Mutant] = field(default_factory=list)
    equivalent: list[tuple[Mutant, str]] = field(default_factory=list)
    no_mutants_reason: str | None = None
    scope: str = "whole file"
    unmutated: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        tested = sum(self.counts.values()) - self.counts["skipped"] - self.counts["equivalent"]
        return None if tested <= 0 else 100.0 * (self.counts["killed"] + self.counts["timeout"]) / tested


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def describe(path: str, keys: list[str], def_line_by_key: dict[str, int]) -> dict[str, tuple[int | None, str, str]]:
    """Map each mutant name to (original line number, original text, mutated text)."""
    module = module_name(path)
    mutant_src = (MUTANTS / path).read_text(encoding="utf-8")
    mutant_lines = mutant_src.splitlines()
    mutant_funcs = _functions(ast.parse(mutant_src))
    out: dict[str, tuple[int | None, str, str]] = {}
    for key in keys:
        mangled = key[len(module) + 1:]
        function_key = mangled.split("__mutmut_")[0]
        orig_copy = mutant_funcs.get(function_key + "__mutmut_orig")
        variant = mutant_funcs.get(mangled)
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
        def_line = def_line_by_key.get(function_key)
        out[key] = (def_line + 1 + i1 if def_line else None, " | ".join(s.strip() for s in before[i1:i2]),
                    " | ".join(s.strip() for s in after[j1:j2]))
    return out


def _function_key(module: str, key: str) -> str:
    return key[len(module) + 1:].split("__mutmut_")[0]


def collect(targets: list[Target], rules: dict) -> list[FileResult]:
    results = []
    for target in targets:
        path = target.path
        source = Path(path).read_text(encoding="utf-8")
        all_units = units(source)
        scoped = all_units if target.functions is None else [u for u in all_units if u.key in target.functions]
        mutable = {u.key for u in scoped if u.mutable}
        result = FileResult(path, scope="whole file" if target.functions is None else ", ".join(sorted(target.functions)),
                            unmutated=sorted(u.key for u in scoped if not u.mutable))
        results.append(result)
        if not scoped:
            result.no_mutants_reason = ("no functions to mutate (constants and imports only)" if target.functions is None
                                        else "changed lines are outside any function (mutmut mutates function bodies only)")
            continue
        if not mutable:
            result.no_mutants_reason = "mutmut skips decorated functions: " + ", ".join(result.unmutated)
            continue
        meta = MUTANTS / f"{path}.meta"
        if not meta.exists():
            continue  # fails in report(): functions to mutate, but no mutmut results
        module = module_name(path)
        codes = json.loads(meta.read_text(encoding="utf-8"))["exit_code_by_key"]
        keys = [k for k in codes if _function_key(module, k) in mutable]
        if not keys:
            continue  # fails in report(): mutmut generated nothing for the scoped functions
        details = describe(path, keys, {u.key: u.def_line for u in all_units})
        facts = SourceFacts.parse(source)
        for key in sorted(keys, key=lambda k: (k.split("__mutmut_")[0], int(k.rsplit("_", 1)[1]))):
            line, original, mutated = details[key]
            mutant = Mutant(key, STATUS.get(codes[key], "suspicious"), line, original, mutated)
            if mutant.status in ("survived", "no tests"):
                reason = equivalence(path, mutant, facts, rules)
                if reason:
                    result.counts["equivalent"] += 1
                    result.equivalent.append((mutant, reason))
                    continue
                result.survivors.append(mutant)
            result.counts[mutant.status] += 1
        if keys and result.score is None:
            result.no_mutants_reason = f"all {len(keys)} mutants are equivalent or skipped"
    return results


def report(results: list[FileResult], threshold: float, *, verbose: bool) -> bool:
    passed = True
    for result in results:
        score = result.score
        if score is None:
            if result.no_mutants_reason:
                print(f"mutation gate: pass: {result.path} no mutants: {result.no_mutants_reason}")
            else:
                passed = False
                print(f"mutation gate: FAIL: {result.path} produced no mutants for its functions "
                      f"({result.scope}); mutmut generated or ran nothing for them")
            continue
        verdict = "pass" if score >= threshold else "FAIL"
        passed &= score >= threshold
        print(f"mutation gate: {verdict}: {result.path} score {score:.1f}% {dict(result.counts)} scope: {result.scope}")
        if result.unmutated:
            print(f"  not mutated (mutmut skips decorated functions): {', '.join(result.unmutated)}")
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
    source.add_argument("--base", help="mutate code changed, or reached by tests changed, versus this ref")
    source.add_argument("--paths", nargs="+", help="mutate these sources instead of a diff")
    parser.add_argument("--functions", nargs="*", help="with --paths: only these mangled functions (x_f, xǁCǁm)")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--no-run", action="store_true", help="score the existing mutants/ results")
    parser.add_argument("--verbose", action="store_true", help="also list equivalent mutants")
    args = parser.parse_args()

    if args.paths:
        functions = frozenset(args.functions) if args.functions else None
        targets = [Target(path, functions) for path in args.paths]
    else:
        targets = changed_targets(args.base)
    if not targets:
        print(f"mutation gate: skipped: no {SOURCE_ROOT} source changed, and no changed or deleted test "
              f"references one, versus {args.base}")
        return 0
    rules = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("tool", {}).get("mutation_gate", {})
    to_run = [t for t in targets
              if any(u.mutable and (t.functions is None or u.key in t.functions)
                     for u in units(Path(t.path).read_text(encoding="utf-8")))]
    if to_run and not args.no_run:
        tests_by_source = {t.path: importing_tests(t.path) for t in to_run}
        untested = [path for path, tests in tests_by_source.items() if not tests]
        if untested:
            print("mutation gate: FAIL: no test file imports", *untested, sep="\n  ")
            return 1
        tests = sorted({t for found in tests_by_source.values() for t in found})
        print("mutation gate: mutating", *(f"{t.path} ({'whole file' if t.functions is None else ', '.join(sorted(t.functions))})"
                                          for t in to_run), sep="\n  ")
        print("mutation gate: with tests", *tests, sep="\n  ")
        text = PYPROJECT.read_text(encoding="utf-8")
        text = set_mutmut_list(text, "only_mutate", [t.path for t in to_run])
        text = set_mutmut_list(text, "pytest_add_cli_args_test_selection", tests)
        PYPROJECT.write_text(text, encoding="utf-8")
        run = subprocess.run(["mutmut", "run", *(g for t in to_run for g in mutant_globs(t))], check=False)
        if run.returncode != 0:
            print(f"mutation gate: FAIL: mutmut run exited {run.returncode}")
            return run.returncode
    ok = report(collect(targets, rules), args.threshold, verbose=args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
