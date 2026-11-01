"""One trial on the ledger: the sealed boundary, the seeds, the lineage stamp and the entry format.

A runner starts a trial with a seal policy, configures it, takes its seeds, produces results and records them.
Every id here must equal what the runners wrote before this module existed, so the formulas are fixed:

- ``seed`` is ``seed_from_config(config)``; a per-test seed is ``seed_from_config({**config, field: key})``,
  with ``field`` "spec" for most runners and "rule" for v7.
- ``trial_id`` is ``short_id`` of the stamp ``{candidate, config, seed, code_version, timestamp}``.
- ``params_hash`` is ``sha`` of whatever the runner names as its parameters: the candidate's ``params`` for a
  bench run, the whole config for the event and exposure studies.

The bench run stamps before it runs (its ``sealed_open`` entry carries the stamp); the studies stamp when they
record, after the work. ``stamp`` is idempotent, so both keep the timestamp they always had.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import ledger

PIN_IF_MISSING = "pin_if_missing"  # the bench run: the first contact writes the seal entry
READ_ONLY = "read_only"  # every study: a ledger without a seal stops the run
POLICIES = (PIN_IF_MISSING, READ_ONLY)
CORE_FIELDS = ("candidate", "config", "seed", "code_version", "timestamp", "event", "trial_id",
               "sealed_opened", "sealed_boundary", "params_hash", "summary")


def read_seal(path: str | Path) -> str:
    """The date pinned by the ledger's seal entry. Read only: a missing seal is an error, not a reason to create one."""
    for e in ledger.read(path):
        if e.get("event") == "seal":
            return e["boundary_date"]
    raise LookupError(f"{path} has no seal entry; the sealed window is not pinned, so nothing is run")


def resolve_boundary(path: str | Path, policy: str, sessions: list[str] | None = None) -> str:
    """The first sealed date under ``policy``; ``PIN_IF_MISSING`` needs the sessions to pin from."""
    if policy == PIN_IF_MISSING:
        if sessions is None:
            raise ValueError("pinning the seal needs the sessions")
        return ledger.pinned_boundary(path, sessions)
    if policy == READ_ONLY:
        return read_seal(path)
    raise ValueError(f"unknown seal policy {policy!r}; expected one of {POLICIES}")


class Trial:
    """One run's lineage. ``start`` resolves the boundary; ``configure`` fixes the config and the seed."""

    def __init__(self, path: str | Path, boundary: str):
        self.path, self.boundary = path, boundary
        self.candidate: str | None = None
        self.config: dict | None = None
        self.seed: int | None = None
        self._stamp: dict | None = None

    def configure(self, candidate: str, config: dict) -> int:
        self.candidate, self.config = candidate, config
        self.seed = ledger.seed_from_config(config)
        return self.seed

    def seed_for(self, key: str, field: str = "spec") -> int:
        return ledger.seed_from_config({**self._configured(), field: key})

    def stamp(self) -> dict:
        """The lineage block every entry of this trial shares; taken once, on first use."""
        if self._stamp is None:
            self._stamp = {"candidate": self.candidate, "config": self._configured(), "seed": self.seed,
                           "code_version": ledger.code_version(), "timestamp": datetime.now(timezone.utc).isoformat()}
        return self._stamp

    @property
    def trial_id(self) -> str:
        return ledger.short_id(self.stamp())

    def open_sealed(self) -> None:
        """Record that this trial reads the sealed window, with how often the candidate already did."""
        prior = ledger.sealed_openings(self.path, self.candidate)
        ledger.append(self.path, {**self.stamp(), "event": "sealed_open", "prior_openings": prior})

    def record(self, event: str, summary: dict, *, params, sealed_opened: bool = False,
               sealed_boundary: str | None = None, **fields) -> str:
        """Append the trial's result entry and return its trial_id. ``fields`` are the event's own extra keys."""
        clash = set(fields) & set(CORE_FIELDS)
        if clash:
            raise ValueError(f"fields {sorted(clash)} are set by the trial, not the runner")
        trial_id = self.trial_id
        ledger.append(self.path, {**self.stamp(), "event": event, "trial_id": trial_id,
                                  "sealed_opened": sealed_opened,
                                  "sealed_boundary": self.boundary if sealed_boundary is None else sealed_boundary,
                                  "params_hash": ledger.sha(params), "summary": summary, **fields})
        return trial_id

    def _configured(self) -> dict:
        if self.config is None:
            raise RuntimeError("configure the trial before taking seeds or stamping it")
        return self.config


def start(path: str | Path, policy: str, sessions: list[str] | None = None) -> Trial:
    return Trial(path, resolve_boundary(path, policy, sessions))
