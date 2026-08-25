from __future__ import annotations

import json
from pathlib import Path

from trader_koo.hyperliquid.routes import _study_evidence


def test_committed_study_is_explicitly_unvalidated_and_non_chronological():
    path = Path("trader_koo/hyperliquid/machibro_study.json")
    payload = json.loads(path.read_text())

    evidence = _study_evidence(payload, source="static_artifact")

    assert evidence["status"] == "unvalidated_retrospective"
    assert evidence["data_end"] == "2026-02-15"
    assert "equity_curve_not_chronological" in evidence["reasons"]
