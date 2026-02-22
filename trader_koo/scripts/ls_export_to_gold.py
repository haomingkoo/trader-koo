from __future__ import annotations

"""
Convert a Label Studio export JSON back to review_decisions.csv
for use with apply_cv_review.py.

Workflow:
    1. Export annotations from Label Studio:
           Project > Export > JSON (not "JSON-MIN")
           Save as e.g. data/cv/ls_annotations.json

    2. Run this script:
           python scripts/ls_export_to_gold.py \
               --ls-export     data/cv/ls_annotations.json \
               --weak-labels   data/cv/weak_labels.csv \
               --out-decisions data/cv/review_decisions.csv

    3. Apply decisions to produce gold labels:
           python scripts/apply_cv_review.py \
               --weak-labels-csv    data/cv/weak_labels.csv \
               --review-decisions-csv data/cv/review_decisions.csv \
               --out-dir            data/cv

Label Studio annotation conventions expected:
    - RectangleLabels named "label" or "pattern" → pattern class
    - Choices named "status"                     → status override (optional)
    - TextArea named "review_notes"              → notes (optional)
    - Skipped/empty tasks → approved = False
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _extract_from_result(result: list[dict]) -> tuple[list[str], str, str]:
    """
    Parse one annotation's result list.
    Returns (patterns, status_override, review_notes).
    """
    patterns: list[str] = []
    status_override = ""
    review_notes = ""

    for item in result:
        item_type = str(item.get("type", "")).lower()
        value = item.get("value", {})
        from_name = str(item.get("from_name", "")).lower()

        if item_type == "rectanglelabels":
            labels = value.get("rectanglelabels", [])
            patterns.extend(str(l) for l in labels if l)

        elif item_type == "choices":
            if "status" in from_name:
                choices = value.get("choices", [])
                if choices:
                    status_override = str(choices[0])

        elif item_type == "textarea":
            if "note" in from_name or "review" in from_name:
                texts = value.get("text", [])
                if texts:
                    review_notes = str(texts[0]).strip()

    return patterns, status_override, review_notes


def _get_reviewer(annotation: dict) -> str:
    """Extract reviewer name/email from annotation metadata."""
    created_by = annotation.get("created_username") or annotation.get("created_ago") or ""
    if isinstance(created_by, dict):
        return str(created_by.get("email") or created_by.get("username") or "")
    return str(created_by)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #

def ls_export_to_decisions(
    ls_export: list[dict],
    weak_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse Label Studio tasks and produce review_decisions rows.

    One row per (sample_id, pattern) pair observed in annotations.
    Skipped/unannotated tasks produce approved=False rows for every
    weak-label pattern in that task.
    """
    weak_df = weak_df.copy()
    weak_df["sample_id"] = weak_df["sample_id"].astype(str)
    weak_by_sid: dict[str, list[str]] = (
        weak_df.groupby("sample_id")["pattern"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )

    rows: list[dict] = []

    for task in ls_export:
        data = task.get("data", {})
        sample_id = str(data.get("sample_id") or task.get("meta", {}).get("sample_id") or "")
        if not sample_id:
            continue

        annotations = task.get("annotations", [])

        # ------------------------------------------------------------------- #
        # Skipped / unannotated task → mark all weak-label patterns rejected
        # ------------------------------------------------------------------- #
        if not annotations:
            for pat in weak_by_sid.get(sample_id, []):
                rows.append({
                    "sample_id": sample_id,
                    "pattern": pat,
                    "approved": False,
                    "label_override": "",
                    "status_override": "",
                    "review_notes": "skipped",
                    "reviewer": "",
                })
            continue

        # ------------------------------------------------------------------- #
        # Use the most recent (last) non-skipped annotation
        # ------------------------------------------------------------------- #
        ann = None
        for a in reversed(annotations):
            if not a.get("was_skipped", False) and a.get("result") is not None:
                ann = a
                break

        if ann is None:
            # All skipped
            for pat in weak_by_sid.get(sample_id, []):
                rows.append({
                    "sample_id": sample_id,
                    "pattern": pat,
                    "approved": False,
                    "label_override": "",
                    "status_override": "",
                    "review_notes": "skipped",
                    "reviewer": "",
                })
            continue

        result = ann.get("result") or []
        reviewer = _get_reviewer(ann)
        labeled_patterns, status_override, review_notes = _extract_from_result(result)

        # ------------------------------------------------------------------- #
        # Map labeled boxes back to weak-label rows
        # Strategy: each labeled pattern name is either:
        #   (a) the same class → approved
        #   (b) a different class → it's a label_override for one weak-label row
        # ------------------------------------------------------------------- #
        original_patterns = weak_by_sid.get(sample_id, [])

        if not labeled_patterns:
            # Annotator cleared all boxes → reject everything
            for pat in original_patterns:
                rows.append({
                    "sample_id": sample_id,
                    "pattern": pat,
                    "approved": False,
                    "label_override": "",
                    "status_override": status_override,
                    "review_notes": review_notes or "all boxes removed",
                    "reviewer": reviewer,
                })
            continue

        # Match each labeled pattern to its original weak-label row
        # (simple best-effort: match same name first, then positional)
        orig_remaining = list(original_patterns)
        labeled_remaining = list(labeled_patterns)

        approved_map: dict[str, dict] = {}

        # Exact-match pass
        for lp in list(labeled_remaining):
            if lp in orig_remaining:
                orig_remaining.remove(lp)
                labeled_remaining.remove(lp)
                approved_map[lp] = {
                    "approved": True,
                    "label_override": "",
                    "status_override": status_override,
                    "review_notes": review_notes,
                    "reviewer": reviewer,
                }

        # Override pass: remaining labeled → override for remaining original
        for lp, op in zip(labeled_remaining, orig_remaining):
            approved_map[op] = {
                "approved": True,
                "label_override": lp,
                "status_override": status_override,
                "review_notes": review_notes,
                "reviewer": reviewer,
            }
            orig_remaining.remove(op)

        # Any original patterns without a matched label → rejected
        for op in orig_remaining:
            approved_map[op] = {
                "approved": False,
                "label_override": "",
                "status_override": status_override,
                "review_notes": review_notes or "box removed",
                "reviewer": reviewer,
            }

        for pat, info in approved_map.items():
            rows.append({"sample_id": sample_id, "pattern": pat, **info})

    df = pd.DataFrame(rows, columns=[
        "sample_id", "pattern", "approved", "label_override",
        "status_override", "review_notes", "reviewer",
    ])
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    export_path = Path(args.ls_export).resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Label Studio export not found: {export_path}")

    weak_path = Path(args.weak_labels).resolve()
    if not weak_path.exists():
        raise FileNotFoundError(f"Weak labels not found: {weak_path}")

    ls_data = json.loads(export_path.read_text(encoding="utf-8"))
    weak_df = pd.read_csv(weak_path)

    decisions = ls_export_to_decisions(ls_data, weak_df)

    out_path = Path(args.out_decisions).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out_path, index=False)

    approved = int(decisions["approved"].astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    rejected = int(len(decisions)) - approved

    print(json.dumps({
        "total_decisions": len(decisions),
        "approved": approved,
        "rejected": rejected,
        "out_decisions": str(out_path),
        "next_step": (
            f"python scripts/apply_cv_review.py "
            f"--weak-labels-csv {args.weak_labels} "
            f"--review-decisions-csv {args.out_decisions} "
            f"--out-dir data/cv"
        ),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Label Studio export JSON to review_decisions.csv."
    )
    p.add_argument("--ls-export", default="data/cv/ls_annotations.json",
                   help="Label Studio JSON export file (Project > Export > JSON).")
    p.add_argument("--weak-labels", default="data/cv/weak_labels.csv",
                   help="Weak labels CSV generated by build_cv_weak_labels.py.")
    p.add_argument("--out-decisions", default="data/cv/review_decisions.csv",
                   help="Output review decisions CSV for apply_cv_review.py.")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
