#!/usr/bin/env python3
"""Run the full pipeline on a single thali image.

Usage:
    python scripts/predict_image.py path/to/plate.jpg
    python scripts/predict_image.py path/to/plate.jpg --classifier convnext
    python scripts/predict_image.py path/to/plate.jpg --classifier ensemble
    python scripts/predict_image.py path/to/plate.jpg --task 3          # force BEV
    python scripts/predict_image.py path/to/plate.jpg --tray-profile none   # force relative-area portion scaling
    python scripts/predict_image.py path/to/plate.jpg --outdir outputs/run1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thali_pipeline.config import load_config
from thali_pipeline.io_utils import save_image_bgr, save_json
from thali_pipeline.pipeline.run import build_context, process_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to a thali image")
    parser.add_argument("--config", default="configs/weights.yaml")
    parser.add_argument(
        "--classifier", default=None, choices=["dino", "convnext", "ensemble"],
        help="default: classifiers.default in the config (currently 'dino')",
    )
    parser.add_argument(
        "--task", default=None, choices=["auto", "2", "3"],
        help="override routing.mode from the config; '2'/'3' forces manual routing",
    )
    parser.add_argument("--outdir", default=None, help="default: output.dir in the config")
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

    result, annotated_bgr = process_image(
        args.image,
        ctx,
        classifier_mode=classifier_mode,
        task_mode=task_mode,
        manual_task=manual_task,
    )

    outdir = Path(args.outdir or config["output"]["dir"])
    stem = Path(args.image).stem
    image_out_path = outdir / f"{stem}_annotated.png"
    json_out_path = outdir / f"{stem}.json"

    save_image_bgr(image_out_path, annotated_bgr)
    save_json(json_out_path, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nAnnotated image -> {image_out_path}")
    print(f"JSON             -> {json_out_path}")


if __name__ == "__main__":
    main()
