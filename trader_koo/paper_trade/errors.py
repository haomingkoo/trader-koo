"""Stable domain errors shared by live admission and promotion replay."""


REPORT_LINEAGE_ERROR_CODES = frozenset({
    "report_not_verified_published",
    "report_publication_lineage_invalid",
    "report_not_current_publication",
})

ADMISSION_PHASE_ERROR_CODES = frozenset({
    "admission_setup_persistence_failed",
    "admission_paper_trade_persistence_failed",
    "admission_finalize_failed",
})

ADMISSION_ERROR_CODES = REPORT_LINEAGE_ERROR_CODES | ADMISSION_PHASE_ERROR_CODES

# A pre-release implementation could write this code. It remains readable in
# immutable ledgers but the current insert contract cannot create it.
LEGACY_ADMISSION_ERROR_CODES = frozenset({"admission_lineage_failed"})
ADMISSION_LEDGER_MIGRATION = "admission-ledger-contract-v5"


class AdmissionLedgerContractError(RuntimeError):
    """Sanitized diagnostics for immutable legacy rows that block migration."""

    def __init__(self, invalid_count: int, attempts: list[dict[str, object]]) -> None:
        self.invalid_count = int(invalid_count)
        self.attempts = attempts
        super().__init__(
            "legacy report admission attempts violate the audit contract: "
            f"invalid_rows={self.invalid_count}"
        )


class ReportLineageError(ValueError):
    """Pre-admission failure for unverifiable report publication."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in REPORT_LINEAGE_ERROR_CODES:
            raise ValueError(f"unknown report-lineage error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}")
