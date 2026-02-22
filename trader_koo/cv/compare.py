from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class HybridCVCompareConfig:
    max_rows: int = 12


COMPARE_COLUMNS = [
    "pattern",
    "hybrid_status",
    "cv_status",
    "hybrid_confidence",
    "cv_confidence",
    "consensus_confidence",
    "agreement",
    "confidence_gap",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COMPARE_COLUMNS)


def compare_hybrid_vs_cv(
    hybrid_patterns: pd.DataFrame,
    cv_patterns: pd.DataFrame,
    cfg: HybridCVCompareConfig,
) -> pd.DataFrame:
    if (hybrid_patterns is None or hybrid_patterns.empty) and (cv_patterns is None or cv_patterns.empty):
        return _empty()

    hp = hybrid_patterns.copy() if hybrid_patterns is not None else pd.DataFrame()
    cp = cv_patterns.copy() if cv_patterns is not None else pd.DataFrame()
    if not hp.empty:
        hp["hybrid_confidence"] = pd.to_numeric(hp.get("hybrid_confidence"), errors="coerce")
        hp = hp.sort_values("hybrid_confidence", ascending=False).drop_duplicates(subset=["pattern"])
        hp = hp[["pattern", "status", "hybrid_confidence"]].rename(columns={"status": "hybrid_status"})
    else:
        hp = pd.DataFrame(columns=["pattern", "hybrid_status", "hybrid_confidence"])

    if not cp.empty:
        cp["cv_confidence"] = pd.to_numeric(cp.get("cv_confidence"), errors="coerce")
        cp = cp.sort_values("cv_confidence", ascending=False).drop_duplicates(subset=["pattern"])
        cp = cp[["pattern", "status", "cv_confidence"]].rename(columns={"status": "cv_status"})
    else:
        cp = pd.DataFrame(columns=["pattern", "cv_status", "cv_confidence"])

    merged = hp.merge(cp, on="pattern", how="outer")
    if merged.empty:
        return _empty()

    merged["hybrid_confidence"] = pd.to_numeric(merged["hybrid_confidence"], errors="coerce")
    merged["cv_confidence"] = pd.to_numeric(merged["cv_confidence"], errors="coerce")
    merged["consensus_confidence"] = merged[["hybrid_confidence", "cv_confidence"]].mean(axis=1, skipna=True)
    merged["confidence_gap"] = (merged["hybrid_confidence"].fillna(0.0) - merged["cv_confidence"].fillna(0.0)).abs()

    def agree_row(r: pd.Series) -> str:
        has_h = pd.notna(r.get("hybrid_confidence"))
        has_c = pd.notna(r.get("cv_confidence"))
        if has_h and has_c:
            if str(r.get("hybrid_status") or "") == str(r.get("cv_status") or ""):
                return "aligned"
            return "pattern_match_status_diff"
        return "single_source"

    merged["agreement"] = merged.apply(agree_row, axis=1)
    merged = merged.sort_values(["consensus_confidence", "agreement"], ascending=[False, True]).head(cfg.max_rows)
    for c in ["hybrid_confidence", "cv_confidence", "consensus_confidence", "confidence_gap"]:
        merged[c] = merged[c].round(2)
    return merged[COMPARE_COLUMNS].reset_index(drop=True)

