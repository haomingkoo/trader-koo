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


class ReportLineageError(ValueError):
    """Pre-admission failure for unverifiable report publication."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in REPORT_LINEAGE_ERROR_CODES:
            raise ValueError(f"unknown report-lineage error code: {code}")
        self.code = code
        super().__init__(f"{code}: {detail}")
