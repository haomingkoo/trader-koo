from __future__ import annotations

"""
Convert cv weak labels to a Label Studio import JSON file.

Setup:
    pip install label-studio
    label-studio start                          # starts on http://localhost:8080

    # Enable local file serving so Label Studio can read images from disk:
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(pwd)/data/cv \\
    label-studio start

Usage:
    python scripts/ls_prepare_tasks.py \\
        --weak-labels-csv data/cv/weak_labels.csv \\
        --images-dir data/cv/images \\
        --out-json data/cv/ls_tasks.json

    # Or if the API is serving images on localhost:8000/cv/images/:
    python scripts/ls_prepare_tasks.py --image-base-url http://localhost:8000/cv/images

Then in Label Studio:
    Project > Import > upload ls_tasks.json

Label config to use (paste into Project Settings > Labeling Interface):
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image">
        <Label value="bull_flag" background="#38d39f"/>
        <Label value="bear_flag" background="#ff6b6b"/>
        <Label value="rising_wedge" background="#ff6b6b"/>
        <Label value="falling_wedge" background="#38d39f"/>
        <Label value="ascending_triangle" background="#38d39f"/>
        <Label value="descending_triangle" background="#ff6b6b"/>
        <Label value="symmetrical_triangle" background="#f8c24e"/>
        <Label value="double_top" background="#ff6b6b"/>
        <Label value="double_bottom" background="#38d39f"/>
        <Label value="head_and_shoulders" background="#ff6b6b"/>
        <Label value="inv_head_and_shoulders" background="#38d39f"/>
        <Label value="cup_and_handle" background="#6aa9ff"/>
      </RectangleLabels>
    </View>
"""

import argparse
import json
import uuid
from pathlib import Path

import pandas as pd


def yolo_to_ls(x_c: float, y_c: float, w: float, h: float) -> dict:
    """Convert YOLO [0-1] center coords to Label Studio top-left [0-100] percent."""
    return {
        "x": max(0.0, min(100.0, (x_c - w / 2) * 100)),
        "y": max(0.0, min(100.0, (y_c - h / 2) * 100)),
        "width": max(0.5, min(100.0, w * 100)),
        "height": max(0.5, min(100.0, h * 100)),
    }


def build_tasks(
    weak_labels: pd.DataFrame,
    images_dir: Path,
    image_base_url: str,
    image_width: int,
    image_height: int,
    only_review: bool,
) -> list[dict]:
    if only_review:
        weak_labels = weak_labels[weak_labels["decision"].isin({"review", "auto_accept"})].copy()

    tasks = []
    for sample_id, group in weak_labels.groupby("sample_id"):
        sample_id = str(sample_id)
        img_name = f"{sample_id}.png"
        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        if image_base_url:
            img_url = f"{image_base_url.rstrip('/')}/{img_name}"
        else:
            # Label Studio local storage format — requires LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
            # pointed at the data/cv/ directory
            img_url = f"/data/local-files/?d=images/{img_name}"

        predictions = []
        for _, row in group.iterrows():
            pattern = str(row.get("pattern") or "")
            if not pattern:
                continue
            conf = float(row.get("consensus_conf") or 0.0)
            xc = float(row.get("yolo_x_center") or 0.5)
            yc = float(row.get("yolo_y_center") or 0.5)
            w = float(row.get("yolo_w") or 0.3)
            h = float(row.get("yolo_h") or 0.3)
            ls_coords = yolo_to_ls(xc, yc, w, h)
            predictions.append({
                "id": str(uuid.uuid4())[:8],
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "value": {**ls_coords, "rotation": 0, "rectanglelabels": [pattern]},
                "score": round(conf, 3),
            })

        if not predictions:
            continue

        mean_conf = float(group["consensus_conf"].mean()) if "consensus_conf" in group.columns else 0.5
        tasks.append({
            "data": {
                "image": img_url,
                "sample_id": sample_id,
                "ticker": str(group["ticker"].iloc[0]) if "ticker" in group.columns else "",
                "end_date": str(group.get("end_date", pd.Series([""])).iloc[0]),
            },
            "predictions": [{
                "model_version": "cv_proxy_v1",
                "score": round(mean_conf, 3),
                "result": predictions,
            }],
            "meta": {"sample_id": sample_id},
        })

    return tasks


def main(args: argparse.Namespace) -> None:
    weak_path = Path(args.weak_labels_csv)
    if not weak_path.exists():
        raise FileNotFoundError(f"Weak labels not found: {weak_path}. Run build_cv_weak_labels.py first.")

    weak_labels = pd.read_csv(weak_path)
    images_dir = Path(args.images_dir)

    tasks = build_tasks(
        weak_labels=weak_labels,
        images_dir=images_dir,
        image_base_url=args.image_base_url,
        image_width=args.image_width,
        image_height=args.image_height,
        only_review=args.only_review,
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    print(json.dumps({
        "tasks_created": len(tasks),
        "images_found": sum(1 for t in tasks if t),
        "out_json": str(out_path),
        "next_step": "Import ls_tasks.json into Label Studio via Project > Import",
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert cv weak labels to Label Studio import JSON.")
    p.add_argument("--weak-labels-csv", default="data/cv/weak_labels.csv")
    p.add_argument("--images-dir", default="data/cv/images")
    p.add_argument("--out-json", default="data/cv/ls_tasks.json")
    p.add_argument(
        "--image-base-url",
        default="",
        help="HTTP base URL for images, e.g. http://localhost:8000/cv/images. "
             "Leave empty to use Label Studio local storage (requires LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT).",
    )
    p.add_argument("--image-width", type=int, default=1280)
    p.add_argument("--image-height", type=int, default=720)
    p.add_argument(
        "--only-review",
        action="store_true",
        help="Only include review/auto_accept samples; skip low_conf.",
    )
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
