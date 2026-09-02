"""Run the portion pipeline over a folder and write a cumulative JSON + CSV report."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

from portion_estimation.factory import build_components
from portion_estimation.pipeline import PortionEstimationPipeline
from portion_estimation.visualization import render_overlay

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FIELDS = [
    "image", "food_index", "label", "physical_type", "compartment_coverage",
    "classification_confidence", "area_cm2", "mean_height_cm", "volume_ml",
    "weight_g", "weight_confidence",
]


def image_paths(folder: Path, recursive: bool, excluded_names: set[str]) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and path.name.lower() not in excluded_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch food-weight estimation for a folder of images.")
    parser.add_argument("--images", default="images", help="Folder containing source images (default: images).")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Pipeline YAML configuration.")
    parser.add_argument("--output", default="outputs/batch_results.json", help="Destination cumulative JSON report.")
    parser.add_argument("--csv-output", default="outputs/batch_results.csv", help="Destination cumulative CSV report (one row per food instance). Pass '' to skip.")
    parser.add_argument("--recursive", action="store_true", help="Also process images in subfolders.")
    parser.add_argument("--visualizations", default="outputs/overlays", help="Folder for grams/class overlay images. Pass '' to skip.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    image_dir = Path(args.images)
    if not image_dir.is_dir():
        raise SystemExit(f"Image folder does not exist: {image_dir}")
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_cfg = config.get("output", {})
    min_label_weight_g = float(output_cfg.get("min_label_weight_g", 3.0))
    # The empty reference is calibration data, never a meal prediction.
    excluded_names = {"empty_tray.jpg"}
    paths = image_paths(image_dir, args.recursive, excluded_names)
    if not paths:
        raise SystemExit(f"No supported meal images found in: {image_dir}")
    pipeline = PortionEstimationPipeline(**build_components(config))
    visualization_dir = Path(args.visualizations) if args.visualizations else None
    if visualization_dir:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    csv_rows: list[dict[str, object]] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            failures.append({"image": str(path), "error": "OpenCV could not read this image."})
            continue
        try:
            result = pipeline.predict(image)
            record = {"image": str(path), "result": result.to_dict()}
            successes.append(record)
            for index, food_row in enumerate(record["result"]["foods"], start=1):
                csv_rows.append({"image": str(path), "food_index": index, **food_row})
            if visualization_dir:
                target = visualization_dir / f"{path.stem}_overlay.png"
                cv2.imwrite(str(target), render_overlay(image, result, min_label_weight_g=min_label_weight_g))
            logging.info("Processed %s: %.1f g", path.name, record["result"]["total_weight_g"])
        except Exception as error:  # Preserve the rest of a long batch if one image fails.
            logging.exception("Failed to process %s", path)
            failures.append({"image": str(path), "error": str(error)})

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config)),
        "images_folder": str(image_dir),
        "summary": {
            "images_found": len(paths),
            "images_processed": len(successes),
            "images_failed": len(failures),
            "total_weight_g": sum(float(item["result"]["total_weight_g"]) for item in successes),
        },
        "images": successes,
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Wrote %s", output)

    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        logging.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
