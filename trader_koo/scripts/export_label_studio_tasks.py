from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def yolo_to_label_studio_rect(row: dict[str, Any], label_name: str, rid: str) -> dict[str, Any]:
    xc = float(row.get("yolo_x_center", 0.5))
    yc = float(row.get("yolo_y_center", 0.5))
    w = float(row.get("yolo_w", 0.4))
    h = float(row.get("yolo_h", 0.4))
    x = clamp01(xc - w / 2.0) * 100.0
    y = clamp01(yc - h / 2.0) * 100.0
    w_pct = clamp01(w) * 100.0
    h_pct = clamp01(h) * 100.0
    return {
        "id": rid,
        "from_name": "pattern",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": round(x, 4),
            "y": round(y, 4),
            "width": round(w_pct, 4),
            "height": round(h_pct, 4),
            "rotation": 0.0,
            "rectanglelabels": [label_name],
        },
    }


def build_label_studio_config(classes: list[str]) -> str:
    labels = "\n".join([f'      <Label value="{c}"/>' for c in classes])
    return f"""<View>
  <Header value="Pattern Review"/>
  <Text name="sample_id_text" value="$sample_id"/>
  <Image name="image" value="$image"/>
  <RectangleLabels name="pattern" toName="image">
{labels}
  </RectangleLabels>
  <Choices name="status" toName="image" choice="single-radio" showInLine="true">
    <Choice value="forming"/>
    <Choice value="breakout"/>
    <Choice value="breakdown"/>
  </Choices>
  <TextArea name="review_notes" toName="image" rows="2" placeholder="Notes (optional)"/>
</View>
"""


def resolve_image_url(image_path: str, out_dir: Path, mode: str, local_files_base: str) -> str:
    if mode == "relative":
        return image_path
    abs_path = str((out_dir / image_path).resolve())
    if mode == "absolute":
        return abs_path
    # mode == "local-files"
    base = local_files_base.rstrip("/")
    # Label Studio local-files backend typically uses /data/local-files/?d=<relative-path-from-storage-root>
    # We pass absolute path when storage root allows it.
    return f"{base}?d={abs_path}"


def run(args: argparse.Namespace) -> None:
    cv_dir = Path(args.cv_dir).resolve()
    out_json = Path(args.tasks_out).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_config = Path(args.config_out).resolve()
    out_config.parent.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(cv_dir / "samples.csv")
    class_map = pd.read_csv(cv_dir / "class_map.csv")
    label_source = "review_queue.csv" if args.source == "review_queue" else "weak_labels.csv"
    labels = pd.read_csv(cv_dir / label_source)

    if samples.empty:
        out_json.write_text("[]\n", encoding="utf-8")
        out_config.write_text(build_label_studio_config(classes=[]), encoding="utf-8")
        print(json.dumps({"tasks": 0, "tasks_out": str(out_json), "config_out": str(out_config)}, indent=2))
        return

    if labels.empty:
        out_json.write_text("[]\n", encoding="utf-8")
        out_config.write_text(build_label_studio_config(classes=sorted(class_map["class_name"].astype(str).tolist())), encoding="utf-8")
        print(json.dumps({"tasks": 0, "tasks_out": str(out_json), "config_out": str(out_config)}, indent=2))
        return

    labels["sample_id"] = labels["sample_id"].astype(str)
    labels["pattern"] = labels["pattern"].astype(str)
    labels["consensus_conf"] = pd.to_numeric(labels.get("consensus_conf"), errors="coerce")

    if "decision" in labels.columns:
        if args.source == "review_queue":
            labels = labels[labels["decision"].astype(str).isin(["review", "auto_accept", "low_conf"])].copy()
        elif not args.include_low_conf:
            labels = labels[labels["decision"].astype(str) != "low_conf"].copy()

    labels = labels[labels["consensus_conf"].fillna(0.0) >= args.min_confidence].copy()
    if labels.empty:
        out_json.write_text("[]\n", encoding="utf-8")
        out_config.write_text(build_label_studio_config(classes=sorted(class_map["class_name"].astype(str).tolist())), encoding="utf-8")
        print(json.dumps({"tasks": 0, "tasks_out": str(out_json), "config_out": str(out_config)}, indent=2))
        return

    if args.max_tasks > 0:
        keep_ids = (
            labels.groupby("sample_id")["consensus_conf"]
            .max()
            .sort_values(ascending=False)
            .head(args.max_tasks)
            .index.tolist()
        )
        labels = labels[labels["sample_id"].isin(keep_ids)].copy()

    sample_by_id = {str(r["sample_id"]): r for r in samples.to_dict(orient="records")}
    tasks = []
    for sid, grp in labels.groupby("sample_id"):
        srow = sample_by_id.get(str(sid))
        if not srow:
            continue
        image_rel = str(srow.get("image_path") or "")
        if not image_rel:
            continue
        image_value = resolve_image_url(image_rel, cv_dir, args.image_mode, args.local_files_base)

        predictions = []
        result = []
        pred_patterns = []
        for i, r in enumerate(grp.to_dict(orient="records"), start=1):
            patt = str(r.get("pattern") or "").strip()
            if not patt:
                continue
            pred_patterns.append(patt)
            result.append(yolo_to_label_studio_rect(r, patt, rid=f"{sid}_{i}"))
        if result:
            predictions.append(
                {
                    "model_version": "weak_label_v1",
                    "score": float(pd.to_numeric(grp["consensus_conf"], errors="coerce").mean()),
                    "result": result,
                }
            )

        task = {
            "data": {
                "image": image_value,
                "sample_id": sid,
                "ticker": str(srow.get("ticker") or ""),
                "end_date": str(srow.get("end_date") or ""),
            },
            "meta": {
                "sample_id": sid,
                "ticker": str(srow.get("ticker") or ""),
                "end_date": str(srow.get("end_date") or ""),
                "predicted_patterns": sorted(set(pred_patterns)),
                "source_file": label_source,
            },
        }
        if predictions:
            task["predictions"] = predictions
        tasks.append(task)

    classes = sorted(class_map["class_name"].astype(str).tolist())
    config_xml = build_label_studio_config(classes=classes)

    out_json.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    out_config.write_text(config_xml, encoding="utf-8")

    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "classes": len(classes),
                "source": label_source,
                "tasks_out": str(out_json),
                "config_out": str(out_config),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export weak-label tasks to Label Studio format.")
    p.add_argument("--cv-dir", default="data/cv")
    p.add_argument("--source", default="review_queue", choices=["review_queue", "weak_labels"])
    p.add_argument("--tasks-out", default="data/cv/label_studio_tasks.json")
    p.add_argument("--config-out", default="data/cv/label_studio_config.xml")
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--max-tasks", type=int, default=2000)
    p.add_argument("--include-low-conf", action="store_true")
    p.add_argument("--image-mode", choices=["relative", "absolute", "local-files"], default="absolute")
    p.add_argument("--local-files-base", default="/data/local-files/")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)

