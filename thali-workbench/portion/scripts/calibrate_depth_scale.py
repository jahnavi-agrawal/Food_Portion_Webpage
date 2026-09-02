"""Optional: tighten MetricMonoDepthEstimator's scale using a handful of
ruler-measured examples. NOT required to use the pipeline — the per-image
self-calibration (floor=0, compartment ceiling=2.5cm) in
`portion_estimation/depth/metric_mono.py` works with zero extra data.

Use this only if you're willing to manually measure real food height (e.g.
with a ruler placed against the compartment) for a small number of examples
— 10-30 is plenty, this is a 2-parameter linear fit, not a network
fine-tune, so it does NOT need one example per dish class.

CSV format (one row per measured food instance):
    image_path,mask_png_path,measured_height_cm
        image_path      - path to the meal photo
        mask_png_path   - single-channel PNG, nonzero where the food is
        measured_height_cm - height you measured with a ruler

Output: a small JSON {"scale": ..., "shift": ...} you can paste into
configs/baseline.yaml under `depth:` as `manual_scale` / `manual_shift`
(MetricMonoDepthEstimator applies them after its own self-calibration,
as a fine correction only).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="CSV of image_path,mask_png_path,measured_height_cm")
    parser.add_argument("--output", default="depth_scale_calibration.json")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from portion_estimation.depth.metric_mono import MetricMonoDepthEstimator

    estimator = MetricMonoDepthEstimator()

    predicted, measured = [], []
    with open(args.csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = cv2.imread(row["image_path"])
            mask = cv2.imread(row["mask_png_path"], cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                print(f"skipping unreadable row: {row}")
                continue
            food_mask = mask > 0
            plate_mask = np.ones(image.shape[:2], dtype=bool)  # unknown here; use whole frame as a rough anchor
            visible = plate_mask & ~food_mask
            height_map = estimator.estimate(image, plate_mask, visible)
            predicted.append(float(np.mean(height_map[food_mask])))
            measured.append(float(row["measured_height_cm"]))

    if len(predicted) < 3:
        raise SystemExit("Need at least 3 usable rows to fit scale/shift.")

    predicted_arr, measured_arr = np.array(predicted), np.array(measured)
    # Robust-ish linear fit: measured = scale * predicted + shift
    A = np.vstack([predicted_arr, np.ones_like(predicted_arr)]).T
    scale, shift = np.linalg.lstsq(A, measured_arr, rcond=None)[0]

    residual = measured_arr - (scale * predicted_arr + shift)
    result = {
        "scale": float(scale),
        "shift": float(shift),
        "n_samples": len(predicted),
        "mae_cm": float(np.mean(np.abs(residual))),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
