"""Verify immutable point-in-time universe snapshots for research execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

SNAPSHOT_TABLE = "universe_membership_snapshots"
MEMBERS_TABLE = "universe_membership_history"
_SNAPSHOT_COLUMNS = {
    "universe_id", "signal_date", "member_count", "members_sha256",
    "source", "source_asof", "evidence_json", "evidence_sha256", "status",
}
_MEMBER_COLUMNS = {"universe_id", "signal_date", "ticker"}
_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class MembershipContractError(ValueError):
    """Stable fail-closed error raised for an invalid membership snapshot."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class VerifiedMembership:
    """A hash-bound membership contract plus exact members at each signal date."""

    universe_id: str
    members_by_date: dict[str, frozenset[str]]
    contract: dict[str, Any]

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset().union(*self.members_by_date.values())

    def members_on(self, signal_date: str) -> frozenset[str]:
        try:
            return self.members_by_date[signal_date]
        except KeyError as exc:
            raise MembershipContractError("point_in_time_membership_date_missing") from exc


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _table_columns(conn: Any, table: str) -> set[str] | None:
    exists = conn.execute(
        "SELECT 1 FROM main.sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone()
    if exists is None:
        return None
    return {str(row[1]) for row in conn.execute(f"PRAGMA main.table_info({table})")}


def _valid_iso_date(value: object) -> bool:
    try:
        dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _valid_iso_timestamp(value: object) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None


def load_verified_membership(
    conn: Any,
    universe_id: str,
    signal_dates: list[str] | tuple[str, ...],
) -> VerifiedMembership:
    """Load and verify one complete materialized snapshot per requested signal date.

    Runtime code never creates or fills these research-only tables. A caller gets
    either one fully verified contract or a stable error; there is no current-
    constituent fallback.
    """
    requested = tuple(sorted(set(str(value) for value in signal_dates)))
    if not universe_id or not requested or any(not _valid_iso_date(value) for value in requested):
        raise MembershipContractError("point_in_time_membership_request_invalid")

    snapshot_columns = _table_columns(conn, SNAPSHOT_TABLE)
    member_columns = _table_columns(conn, MEMBERS_TABLE)
    if snapshot_columns is None or member_columns is None:
        raise MembershipContractError("point_in_time_membership_schema_unavailable")
    if not _SNAPSHOT_COLUMNS.issubset(snapshot_columns) or not _MEMBER_COLUMNS.issubset(member_columns):
        raise MembershipContractError("point_in_time_membership_schema_invalid")

    placeholders = ",".join("?" for _ in requested)
    snapshot_rows = conn.execute(
        f"""SELECT signal_date,member_count,members_sha256,source,source_asof,
                   evidence_json,evidence_sha256,status
              FROM main.{SNAPSHOT_TABLE}
             WHERE universe_id=? AND signal_date IN ({placeholders})
             ORDER BY signal_date""",
        (universe_id, *requested),
    ).fetchall()
    snapshots = {str(row[0]): row for row in snapshot_rows}
    if len(snapshot_rows) != len(requested) or set(snapshots) != set(requested):
        raise MembershipContractError("point_in_time_membership_date_missing")

    member_rows = conn.execute(
        f"""SELECT signal_date,ticker FROM main.{MEMBERS_TABLE}
             WHERE universe_id=? AND signal_date IN ({placeholders})
             ORDER BY signal_date,ticker""",
        (universe_id, *requested),
    ).fetchall()
    grouped: dict[str, list[str]] = {date: [] for date in requested}
    for date_value, ticker_value in member_rows:
        date = str(date_value)
        ticker = str(ticker_value or "")
        if date not in grouped or ticker != ticker.strip().upper() or not _TICKER.fullmatch(ticker):
            raise MembershipContractError("point_in_time_membership_row_invalid")
        grouped[date].append(ticker)

    manifest: list[dict[str, Any]] = []
    members_by_date: dict[str, frozenset[str]] = {}
    for signal_date in requested:
        row = snapshots[signal_date]
        members = grouped[signal_date]
        if not members:
            raise MembershipContractError("point_in_time_membership_empty")
        if len(members) != len(set(members)):
            raise MembershipContractError("point_in_time_membership_duplicate")
        try:
            member_count = int(row[1])
        except (TypeError, ValueError) as exc:
            raise MembershipContractError("point_in_time_membership_manifest_invalid") from exc
        members_sha256 = str(row[2] or "")
        source = str(row[3] or "").strip()
        source_asof = str(row[4] or "")
        evidence_json = str(row[5] or "")
        evidence_sha256 = str(row[6] or "")
        status = str(row[7] or "")
        try:
            evidence = json.loads(evidence_json)
        except (TypeError, ValueError):
            evidence = None
        expected_evidence = {
            "schema_version": "point-in-time-membership-evidence-v1",
            "universe_id": universe_id,
            "signal_date": signal_date,
            "member_count": member_count,
            "members_sha256": members_sha256,
            "source": source,
            "source_asof": source_asof,
            "retrieved_at": evidence.get("retrieved_at") if isinstance(evidence, dict) else None,
            "artifact_sha256": evidence.get("artifact_sha256") if isinstance(evidence, dict) else None,
        }
        if (
            status != "verified"
            or not source
            or not _valid_iso_date(source_asof)
            or source_asof > signal_date
            or not _SHA256.fullmatch(evidence_sha256)
            or member_count != len(members)
            or members_sha256 != _canonical_sha256(sorted(members))
            or evidence != expected_evidence
            or not _valid_iso_timestamp(expected_evidence["retrieved_at"])
            or not _SHA256.fullmatch(str(expected_evidence["artifact_sha256"] or ""))
            or evidence_sha256 != _canonical_sha256(evidence)
        ):
            raise MembershipContractError("point_in_time_membership_manifest_invalid")
        members_by_date[signal_date] = frozenset(members)
        manifest.append({
            "signal_date": signal_date,
            "member_count": member_count,
            "members_sha256": members_sha256,
            "source": source,
            "source_asof": source_asof,
            "evidence_sha256": evidence_sha256,
        })

    contract_body = {
        "schema_version": "point-in-time-membership-v1",
        "universe_id": universe_id,
        "signal_date_count": len(requested),
        "member_row_count": sum(len(value) for value in members_by_date.values()),
        "signal_start": requested[0],
        "signal_end": requested[-1],
        "snapshots": manifest,
    }
    contract = {
        **contract_body,
        "membership_sha256": _canonical_sha256(contract_body),
    }
    return VerifiedMembership(universe_id, members_by_date, contract)
