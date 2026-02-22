from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_bool(x: object) -> bool:
    s = str(x or "").strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def run(args: argparse.Namespace) -> None:
    weak_path = Path(args.weak_labels_csv).resolve()
    decision_path = Path(args.review_decisions_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    weak = pd.read_csv(weak_path)
    if weak.empty:
        (out_dir / "gold_labels.csv").write_text("", encoding="utf-8")
        print(json.dumps({"gold_labels": 0, "rejected": 0, "pending": 0, "out_dir": str(out_dir)}, indent=2))
        return

    decisions = pd.read_csv(decision_path) if decision_path.exists() else pd.DataFrame(columns=["sample_id", "pattern"])
    for c in ["sample_id", "pattern"]:
        if c not in decisions.columns:
            decisions[c] = ""
    decisions["sample_id"] = decisions["sample_id"].astype(str)
    decisions["pattern"] = decisions["pattern"].astype(str)

    merge_cols = ["sample_id", "pattern"]
    use_cols = ["approved", "label_override", "status_override", "review_notes", "reviewer"]
    for c in use_cols:
        if c not in decisions.columns:
            decisions[c] = ""
    d = decisions[merge_cols + use_cols].copy()
    d = d.drop_duplicates(subset=merge_cols, keep="last")

    weak["sample_id"] = weak["sample_id"].astype(str)
    weak["pattern"] = weak["pattern"].astype(str)
    merged = weak.merge(d, on=merge_cols, how="left")
    merged["approved_flag"] = merged["approved"].map(parse_bool)

    auto_ok = merged["decision"].astype(str).eq("auto_accept")
    manual_ok = merged["approved_flag"]
    keep = auto_ok | manual_ok

    gold = merged[keep].copy()
    if not gold.empty:
        gold["pattern_final"] = gold["pattern"]
        has_override = gold["label_override"].astype(str).str.strip().ne("")
        gold.loc[has_override, "pattern_final"] = gold.loc[has_override, "label_override"].astype(str).str.strip()

        gold["status_final"] = gold["status"]
        has_status = gold["status_override"].astype(str).str.strip().ne("")
        gold.loc[has_status, "status_final"] = gold.loc[has_status, "status_override"].astype(str).str.strip()

        gold["label_source"] = "auto_accept"
        gold.loc[manual_ok[keep].values, "label_source"] = "human_review"
        gold["review_state"] = "confirmed"

    rejected = merged[(merged["decision"].astype(str).eq("review")) & (~merged["approved_flag"])].copy()
    pending = merged[(merged["decision"].astype(str).eq("review")) & (merged["approved"].isna() | merged["approved"].astype(str).eq(""))].copy()

    gold_cols = [
        "sample_id",
        "ticker",
        "pattern_final",
        "status_final",
        "consensus_conf",
        "rule_conf",
        "hybrid_conf",
        "cv_conf",
        "decision",
        "agreement",
        "label_source",
        "review_state",
        "review_notes",
        "reviewer",
        "yolo_x_center",
        "yolo_y_center",
        "yolo_w",
        "yolo_h",
    ]
    for c in gold_cols:
        if c not in gold.columns:
            gold[c] = ""

    gold = gold[gold_cols].rename(columns={"pattern_final": "pattern", "status_final": "status"})
    gold.to_csv(out_dir / "gold_labels.csv", index=False)
    rejected.to_csv(out_dir / "rejected_labels.csv", index=False)
    pending.to_csv(out_dir / "pending_review.csv", index=False)

    print(
        json.dumps(
            {
                "gold_labels": int(len(gold)),
                "rejected": int(len(rejected)),
                "pending": int(len(pending)),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply human review decisions to weak labels and produce gold labels.")
    p.add_argument("--weak-labels-csv", default="data/cv/weak_labels.csv")
    p.add_argument("--review-decisions-csv", default="data/cv/review_decisions.csv")
    p.add_argument("--out-dir", default="data/cv")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
