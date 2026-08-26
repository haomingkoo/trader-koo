"""Authorize the exact price cohort sealed into a canonical report."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from trader_koo.db.price_contract import (
    research_price_contract,
    valid_price_provenance,
)

PRICE_CONTRACT_ERROR_CODES = frozenset({
    "price_contract_missing",
    "price_contract_malformed",
    "price_contract_cohort_missing",
    "price_contract_cohort_malformed",
    "price_contract_execution_cohort_missing",
    "price_contract_revision_malformed",
    "price_contract_cohort_ineligible",
    "price_contract_changed",
})
_HASH_FIELDS = ("price_sha256", "action_sha256", "evidence_sha256", "revision")


class PriceContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        if code not in PRICE_CONTRACT_ERROR_CODES:
            raise ValueError(f"unknown price-contract error code: {code}")
        super().__init__(message)
        self.code = code


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def require_expected_price_contract(
    conn: sqlite3.Connection,
    expected: object,
    *,
    required_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Recompute and authorize only the report's signed ticker cohort."""
    if not isinstance(expected, dict):
        raise PriceContractError("price_contract_missing", "canonical report price contract is missing")
    revision = str(expected.get("revision") or "").strip()
    ticker_contracts = expected.get("ticker_contracts")
    if (
        expected.get("eligible") is not True
        or expected.get("status") != "verified"
        or not _is_sha256(revision)
        or not valid_price_provenance(expected.get("basis"), expected.get("version"))
    ):
        raise PriceContractError("price_contract_malformed", "canonical report price contract is malformed")
    if not isinstance(ticker_contracts, dict) or not ticker_contracts:
        raise PriceContractError(
            "price_contract_cohort_missing",
            "canonical report price contract ticker cohort is missing",
        )

    nested_revisions: dict[str, str] = {}
    for ticker, sealed in ticker_contracts.items():
        if (
            not isinstance(ticker, str)
            or ticker != ticker.strip().upper()
            or not ticker
            or not isinstance(sealed, dict)
            or sealed.get("eligible") is not True
            or not valid_price_provenance(sealed.get("basis"), sealed.get("version"))
            or any(not _is_sha256(sealed.get(field)) for field in _HASH_FIELDS)
        ):
            raise PriceContractError(
                "price_contract_cohort_malformed",
                "canonical report price contract ticker cohort is malformed",
            )
        nested_revisions[ticker] = str(sealed["revision"])

    required: set[str] = set()
    for ticker in required_tickers or []:
        if not isinstance(ticker, str) or ticker != ticker.strip().upper() or not ticker:
            raise PriceContractError(
                "price_contract_execution_cohort_missing",
                "required execution ticker is malformed",
            )
        required.add(ticker)
    missing = sorted(required - set(nested_revisions))
    if missing:
        raise PriceContractError(
            "price_contract_execution_cohort_missing",
            f"{missing[0]} is not in the canonical price contract cohort",
        )
    if _sha256(nested_revisions) != revision:
        raise PriceContractError(
            "price_contract_revision_malformed",
            "canonical report price contract revision does not match its ticker cohort",
        )

    current = research_price_contract(conn, sorted(nested_revisions))
    if not current.get("eligible"):
        raise PriceContractError(
            "price_contract_cohort_ineligible",
            "canonical report price contract ticker cohort is not research eligible",
        )
    if (
        current.get("revision") != revision
        or current.get("basis") != expected.get("basis")
        or current.get("version") != expected.get("version")
    ):
        raise PriceContractError("price_contract_changed", "canonical report price contract changed")
    comparable = (
        "eligible", "basis", "version", "managed_start", "managed_end", "row_count",
        *_HASH_FIELDS,
    )
    for ticker, sealed in ticker_contracts.items():
        observed = current["ticker_contracts"][ticker]
        if any(sealed.get(field) != observed.get(field) for field in comparable):
            raise PriceContractError(
                "price_contract_changed",
                f"canonical report price contract changed for {ticker}",
            )
    return current
