"""Ratcheting guardrails for the backend: file size, bare numbers, duplicate code.

Legacy violations are recorded in ``ops/guardrails_baseline.json`` and may only
shrink. New files start clean. When a file moves, rename its baseline entry in
the same PR so it keeps its allowance without gaining a new one.

    python -m ops.guardrails check      # fail on any new violation
    python -m ops.guardrails baseline   # lower the baseline; seed it if absent
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

LINE_LIMIT = 400
ALLOWED_NUMBERS = frozenset({0, 1, -1, 2, 100})
BASELINE_PATH = Path(__file__).with_name("guardrails_baseline.json")
SKIP_DIRS = frozenset({"tests", "frontend-v2", "node_modules", "__pycache__"})
# Tunables live in config/ and paper/ (today: paper_trade/, config.py, paper_trades.py).
NUMBER_EXEMPT_PARTS = frozenset({"config", "paper", "paper_trade", "config.py", "paper_trades.py"})
DUPLICATE_CMD = [
    sys.executable, "-m", "pylint", "--disable=all", "--enable=duplicate-code",
    "--min-similarity-lines=12", "--output-format=json", "trader_koo",
]

Counts = dict[str, dict[str, int]]


def _named_constant_ids(tree: ast.Module) -> set[int]:
    """Literals that are the value of a module-level UPPER_CASE constant: named, so not magic."""
    named: set[int] = set()
    for stmt in tree.body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        if targets and stmt.value is not None and all(isinstance(t, ast.Name) and t.id.isupper() for t in targets):
            named.update(id(node) for node in ast.walk(stmt.value))
    return named


def _bare_numbers(source: str) -> int:
    tree = ast.parse(source)
    named = _named_constant_ids(tree)
    count = 0
    for node in ast.walk(tree):
        if (
            id(node) not in named
            and isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and node.value not in ALLOWED_NUMBERS
        ):
            count += 1
    return count


def scan(root: Path) -> Counts:
    out: Counts = {}
    for path in sorted((root / "trader_koo").rglob("*.py")):
        rel = path.relative_to(root)
        if SKIP_DIRS & set(rel.parts):
            continue
        source = path.read_text(encoding="utf-8")
        exempt = bool(NUMBER_EXEMPT_PARTS & set(rel.parts))
        out[rel.as_posix()] = {
            "lines": len(source.splitlines()),
            "bare_numbers": 0 if exempt else _bare_numbers(source),
        }
    return out


def check(current: Counts, baseline: Counts, limit: int = LINE_LIMIT) -> list[str]:
    violations: list[str] = []
    for name in sorted(current):
        cur, base = current[name], baseline.get(name, {})
        base_lines = base.get("lines", 0)
        if base_lines > limit:
            if cur["lines"] > base_lines:
                violations.append(f"{name}: {cur['lines']} lines > baseline {base_lines}")
        elif cur["lines"] > limit:
            violations.append(f"{name}: {cur['lines']} lines > {limit}")
        base_numbers = base.get("bare_numbers", 0)
        if cur["bare_numbers"] > base_numbers:
            violations.append(f"{name}: {cur['bare_numbers']} bare numbers > baseline {base_numbers}")
    return violations


def seed_baseline(current: Counts, limit: int = LINE_LIMIT) -> Counts:
    return {
        name: dict(vals)
        for name, vals in current.items()
        if vals["lines"] > limit or vals["bare_numbers"] > 0
    }


def update_baseline(current: Counts, baseline: Counts) -> tuple[Counts, list[str]]:
    """Lower baseline entries to current values. Rising values are reported, never written."""
    new: Counts = {}
    raised: list[str] = []
    for name, base in sorted(baseline.items()):
        cur = current.get(name)
        if cur is None:
            continue
        entry: dict[str, int] = {}
        for key in ("lines", "bare_numbers"):
            if cur[key] > base[key]:
                raised.append(f"{name}: {key} {cur[key]} > baseline {base[key]}")
            entry[key] = min(cur[key], base[key])
        new[name] = entry
    return new, raised


def duplicate_blocks(root: Path) -> int:
    proc = subprocess.run(DUPLICATE_CMD, cwd=root, capture_output=True, text=True, check=False)
    try:
        messages = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"guardrails: pylint produced unreadable output ({exc}): {proc.stderr[-300:]}")
    return sum(1 for m in messages if m.get("symbol") == "duplicate-code")


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ops.guardrails")
    parser.add_argument("cmd", choices=("check", "baseline"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    args = parser.parse_args(argv)

    current = scan(args.root)
    stored = _load(args.baseline)
    dupes = duplicate_blocks(args.root)

    if args.cmd == "check":
        violations = check(current, stored.get("files", {}))
        base_dupes = stored.get("duplicate_blocks", 0)
        if dupes > base_dupes:
            violations.append(f"duplicate code: {dupes} blocks > baseline {base_dupes}")
        print("\n".join(violations) if violations else "guardrails: 0 new violations")
        return 1 if violations else 0

    if not stored:
        _write(args.baseline, {"files": seed_baseline(current), "duplicate_blocks": dupes})
        print(f"guardrails: seeded baseline at {args.baseline}")
        return 0
    files, raised = update_baseline(current, stored.get("files", {}))
    if dupes > stored.get("duplicate_blocks", 0):
        raised.append(f"duplicate code: {dupes} blocks > baseline {stored.get('duplicate_blocks', 0)}")
    _write(args.baseline, {"files": files, "duplicate_blocks": min(dupes, stored.get("duplicate_blocks", 0))})
    print("\n".join(raised) if raised else "guardrails: baseline lowered")
    return 1 if raised else 0


if __name__ == "__main__":
    sys.exit(main())
