#!/usr/bin/env python3
"""Diagnose a YOLO detector that's returning zero boxes.

Bypasses our YoloDishDetector wrapper entirely and talks to ultralytics
directly, at a near-zero confidence threshold, so you can see:
  - whether the checkpoint loads with sane class names at all
  - the RAW highest confidence found on this image, before any threshold
    filtering
  - a saved visualization (via ultralytics' own plotting) so you can look
    at what -- if anything -- it's seeing

Usage:
    python scripts/debug_yolo.py path/to/image.jpg
    python scripts/debug_yolo.py path/to/image.jpg --weights weights/yolo_best.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thali_pipeline.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights", default=None, help="default: yolo.weights from configs/weights.yaml")
    parser.add_argument("--config", default="configs/weights.yaml")
    parser.add_argument("--conf", type=float, default=0.001, help="near-zero, to see raw output")
    args = parser.parse_args()

    config = load_config(args.config)
    weights_path = args.weights or config["yolo"]["weights"]

    print(f"Loading YOLO weights: {weights_path}")
    from ultralytics import YOLO
    model = YOLO(weights_path)

    print(f"\nmodel.task: {model.task}")
    print(f"model.names ({len(model.names)} classes): {model.names}")

    print(f"\nImage: {args.image}")
    import cv2
    img = cv2.imread(args.image)
    if img is None:
        print(f"!! cv2.imread returned None -- bad path or unsupported format: {args.image}")
        return
    print(f"Image shape (H, W, C): {img.shape}")

    print(f"\nRunning YOLO at conf={args.conf} (near-zero, to bypass thresholding)...")
    results = model.predict(source=img, conf=args.conf, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("\n!! ZERO boxes even at conf=0.001.")
        print("   This means the model itself found nothing on this image, not a")
        print("   thresholding issue. Either:")
        print("     - this checkpoint isn't the trained dish detector you expect")
        print("       (double check model.names above looks right), or")
        print("     - this image is genuinely out of distribution for the direct")
        print("       detector (e.g. off-angle) -- which is exactly what Task 3's")
        print("       BEV fallback exists for.")
    else:
        confs = sorted([float(b.conf[0]) for b in boxes], reverse=True)
        print(f"\n{len(boxes)} raw box(es) found. Confidences: {[round(c, 4) for c in confs]}")
        print(f"Highest confidence: {confs[0]:.4f}")
        configured_conf = config["yolo"].get("conf_threshold", 0.10)
        if confs[0] < configured_conf:
            print(f"\n!! Highest confidence ({confs[0]:.4f}) is BELOW your configured "
                  f"yolo.conf_threshold ({configured_conf}) in configs/weights.yaml.")
            print(f"   That's why Task 2 found 0 detections. Try lowering conf_threshold, "
                  f"e.g. to {max(0.02, round(confs[0] * 0.7, 3))}.")
        else:
            print(f"\nHighest confidence clears your configured threshold "
                  f"({configured_conf}) -- if the real pipeline still finds 0, "
                  f"check iou_threshold/agnostic_nms next.")

    out_path = Path("outputs") / f"debug_yolo_{Path(args.image).stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results[0].save(filename=str(out_path))
    print(f"\nVisualization saved -> {out_path}  (open this to see what, if anything, it drew)")


if __name__ == "__main__":
    main()
