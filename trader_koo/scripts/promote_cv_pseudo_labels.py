from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run(args: argparse.Namespace) -> None:
    pred_path = Path(args.model_predictions_csv).resolve()
    gold_path = Path(args.gold_labels_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(pred_path)
    if preds.empty:
        print(json.dumps({"promoted": 0, "review_queue": 0, "out_dir": str(out_dir)}, indent=2))
        return

    required = ["sample_id", "pattern", "confidence"]
    for c in required:
        if c not in preds.columns:
            raise ValueError(f"Missing required column in predictions: {c}")
    if "status" not in preds.columns:
        preds["status"] = "forming"

    preds["sample_id"] = preds["sample_id"].astype(str)
    preds["pattern"] = preds["pattern"].astype(str)
    preds["confidence"] = pd.to_numeric(preds["confidence"], errors="coerce")
    preds = preds.dropna(subset=["confidence"])
    preds = preds[preds["confidence"] >= args.min_confidence].copy()
    if preds.empty:
        print(json.dumps({"promoted": 0, "review_queue": 0, "out_dir": str(out_dir)}, indent=2))
        return

    if args.max_per_pattern > 0:
        preds = preds.sort_values("confidence", ascending=False).groupby("pattern", as_index=False).head(args.max_per_pattern)

    if gold_path.exists():
        gold = pd.read_csv(gold_path)
    else:
        gold = pd.DataFrame(columns=["sample_id", "pattern"])
    for c in ["sample_id", "pattern"]:
        if c not in gold.columns:
            gold[c] = ""
    gold["sample_id"] = gold["sample_id"].astype(str)
    gold["pattern"] = gold["pattern"].astype(str)

    merged = preds.merge(gold[["sample_id", "pattern"]].drop_duplicates(), on=["sample_id", "pattern"], how="left", indicator=True)
    novel = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).copy()

    promoted = novel[novel["confidence"] >= args.promote_threshold].copy()
    promoted["label_source"] = "model_pseudo"
    promoted["review_state"] = "auto_promoted"

    review = novel[(novel["confidence"] >= args.review_threshold) & (novel["confidence"] < args.promote_threshold)].copy()
    review["label_source"] = "model_pseudo"
    review["review_state"] = "needs_human_review"

    promoted.to_csv(out_dir / "pseudo_promoted.csv", index=False)
    review.to_csv(out_dir / "pseudo_review_queue.csv", index=False)

    print(
        json.dumps(
            {
                "promoted": int(len(promoted)),
                "review_queue": int(len(review)),
                "out_dir": str(out_dir),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Promote high-confidence model predictions as pseudo labels.")
    p.add_argument("--model-predictions-csv", default="data/cv/model_predictions.csv")
    p.add_argument("--gold-labels-csv", default="data/cv/gold_labels.csv")
    p.add_argument("--out-dir", default="data/cv")
    p.add_argument("--min-confidence", type=float, default=0.75)
    p.add_argument("--review-threshold", type=float, default=0.85)
    p.add_argument("--promote-threshold", type=float, default=0.93)
    p.add_argument("--max-per-pattern", type=int, default=300)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
