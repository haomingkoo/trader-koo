"""Append-only trial ledger, sealed-window boundary, seeds and code version.

Every run is one JSON line. The entry holds the full config, so ``replay`` can
re-run it; the random-control seed is derived from the config, so the same entry
always yields the same numbers. Every entry also records the sha256 of
``config/research.toml`` as ``settings_sha256``.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from trader_koo.research.settings import get as _settings

SEAL_FRACTION = _settings().bench.seal_fraction
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


class CodeVersionError(RuntimeError):
    pass


def code_version() -> str:
    """Repository HEAD of the code, ``-dirty`` when tracked files changed. Raises when it cannot be read.

    A ledger entry without a code version would break lineage, so there is no placeholder.
    """
    root = Path(__file__).resolve().parents[REPO_ROOT_DEPTH]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CodeVersionError(f"cannot read the version of the code in {root}: {exc}") from exc
    return head + ("-dirty" if dirty else "")


def seal_boundary(n_sessions: int) -> int:
    """First sealed session index of the last 20% of sessions; used once, to pin the date."""
    return int(n_sessions * (1 - SEAL_FRACTION))


def pinned_boundary(path: str | Path, sessions: list[str]) -> str:
    """Date of the first sealed session, fixed by the first ledger contact and never recomputed.

    Deriving the boundary from the current session count would slide it forward
    every time data is appended, pulling formerly sealed days into development.
    The first call writes a ``seal`` entry; later calls (and replays) read it back.
    """
    for e in read(path):
        if e.get("event") == "seal":
            if e["boundary_date"] > sessions[-1]:
                raise ValueError(f"ledger seals {e['boundary_date']} but the data ends {sessions[-1]}; wrong ledger?")
            return e["boundary_date"]
    boundary = sessions[seal_boundary(len(sessions))]
    append(path, {"event": "seal", "boundary_date": boundary, "n_sessions": len(sessions),
                  "first_session": sessions[0], "last_session": sessions[-1],
                  "seal_fraction": SEAL_FRACTION, "timestamp": datetime.now(timezone.utc).isoformat()})
    return boundary


def append(path: str | Path, entry: dict) -> None:
    """Append one entry, stamped with the sha256 of the settings file in force."""
    stamped = {**entry, "settings_sha256": _settings().sha256}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(canonical(stamped) + "\n")


def read(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def sealed_openings(path: str | Path, candidate: str) -> int:
    return sum(1 for e in read(path) if e.get("candidate") == candidate and e.get("event") == "sealed_open")
