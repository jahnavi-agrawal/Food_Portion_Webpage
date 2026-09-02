#!/usr/bin/env python3
"""Batch-run the pipeline over every image in a folder.

Usage:
    python scripts/predict_folder.py path/to/thali_images/
    python scripts/predict_folder.py path/to/thali_images/ --classifier convnext --outdir outputs/batch1

Outputs (written to --outdir):
    <stem>_annotated.png   per image
    <stem>.json            per image
    results.json           combined list, one entry per image
    detections.csv         one row per detected dish across all images
    summary.csv            one row per image (totals)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thali_pipeline.config import load_config
from thali_pipeline.io_utils import save_image_bgr, save_json, write_csv_rows
from thali_pipeline.pipeline.run import build_context, process_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder of thali images")
    parser.add_argument("--config", default="configs/weights.yaml")
    parser.add_argument(
        "--classifier", default=None, choices=["dino", "convnext", "ensemble"]
    )
    parser.add_argument("--task", default=None, choices=["auto", "2", "3"])
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--tray-profile", default=None,
        help="tray profile id under tray_profiles.directory (default: tray_profiles.active in the "
             "config), or 'none' to force relative-area portion scaling regardless of config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    classifier_mode = args.classifier or config["classifiers"].get("default", "dino")
    classifier_modes_to_load = ["dino", "convnext"] if classifier_mode == "ensemble" else [classifier_mode]
    ctx = build_context(config, classifier_modes=classifier_modes_to_load, tray_profile_override=args.tray_profile)

    task_mode, manual_task = None, None
    if args.task in ("2", "3"):
        task_mode, manual_task = "manual", int(args.task)
    elif args.task == "auto":
        task_mode = "auto"

    folder = Path(args.folder)
    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"No images found in {folder}")
        return

    outdir = Path(args.outdir or config["output"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    detection_rows = []
    summary_rows = []

    for path in image_paths:
        print(f"\n=== {path.name} ===")
        try:
            result, annotated_bgr = process_image(
                str(path),
                ctx,
                classifier_mode=classifier_mode,
                task_mode=task_mode,
                manual_task=manual_task,
            )
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        save_image_bgr(outdir / f"{path.stem}_annotated.png", annotated_bgr)
        save_json(outdir / f"{path.stem}.json", result)
        all_results.append(result)

        for det in result["detections"]:
            detection_rows.append({
                "image": result["image"],
                "task_used": result["task_used"],
                "detection_id": det["id"],
                "class": det["class"],
                "class_confidence": det["class_confidence"],
                "detector_confidence": det["detector_confidence"],
                "x1": det["bbox_xyxy"][0], "y1": det["bbox_xyxy"][1],
                "x2": det["bbox_xyxy"][2], "y2": det["bbox_xyxy"][3],
                "calories": det["calories"],
                "calories_clip_raw": det.get("calories_clip_raw"),
                "calories_lookup_estimate": det.get("calories_lookup_estimate"),
                "calories_blend_factor": det.get("calories_blend_factor"),
                "portion_scale_factor": det.get("portion_scale_factor"),
                "portion_scale_source": det.get("portion_scale_source"),
                "tray_compartment": det.get("tray_compartment"),
            })

        summary_rows.append({
            "image": result["image"],
            "task_used": result["task_used"],
            "classifier_mode": result["classifier_mode"],
            "tray_profile": result.get("tray_profile"),
            "num_dishes": result["num_dishes"],
            "total_calories_summed": result["total_calories_summed"],
        })

        print(f"  task={result['task_used']} dishes={result['num_dishes']} "
              f"sum_kcal={result['total_calories_summed']}")

    save_json(outdir / "results.json", {"images": all_results})
    if detection_rows:
        write_csv_rows(
            outdir / "detections.csv", detection_rows,
            fieldnames=list(detection_rows[0].keys()),
        )
    if summary_rows:
        write_csv_rows(
            outdir / "summary.csv", summary_rows,
            fieldnames=list(summary_rows[0].keys()),
        )

    print(f"\nProcessed {len(all_results)}/{len(image_paths)} images.")
    print(f"Outputs -> {outdir}")


if __name__ == "__main__":
    main()
