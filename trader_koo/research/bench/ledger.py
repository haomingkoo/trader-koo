"""Append-only trial ledger, sealed-window boundary, seeds and code version.

Every run is one JSON line. The entry holds the full config, so ``replay`` can
re-run it; the random-control seed is derived from the config, so the same entry
always yields the same numbers.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SEAL_FRACTION = 0.2
HEX_BASE = 16
SEED_HEX_DIGITS = 8
ID_LENGTH = 12
REPO_ROOT_DEPTH = 3


class SealedWindowError(RuntimeError):
    pass


def short_id(obj) -> str:
    return sha(obj)[:ID_LENGTH]


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def seed_from_config(config: dict) -> int:
    return int(sha(config)[:SEED_HEX_DIGITS], HEX_BASE)


def code_version() -> str:
    root = Path(__file__).resolve().parents[REPO_ROOT_DEPTH]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


def seal_boundary(n_sessions: int) -> int:
    """First sealed session index: the last 20% of all sessions in the database."""
    return int(n_sessions * (1 - SEAL_FRACTION))


def append(path: str | Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(canonical(entry) + "\n")


def read(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def sealed_openings(path: str | Path, candidate: str) -> int:
    return sum(1 for e in read(path) if e.get("candidate") == candidate and e.get("event") == "sealed_open")
