"""Stable domain errors shared by live admission and promotion replay."""


class ReportLineageError(ValueError):
    """Pre-admission failure for unverifiable report publication."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")
