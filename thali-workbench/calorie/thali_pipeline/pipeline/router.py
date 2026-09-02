"""Decide whether an image gets Task 2 (direct top-down detection) or
Task 3 (SAM2 tray segmentation -> BEV homography -> CLAHE -> same detection).

Modes (configs/weights.yaml: routing.mode, or override per-call):
  auto       -- try Task 2 first; if it finds fewer than
                `min_direct_detections` boxes, fall back to Task 3. This is
                the default per your instructions ("try direct first, fall
                back to BEV if YOLO finds little/nothing").
  bev_first  -- always run Task 3.
  manual     -- caller must pass task=2 or task=3 explicitly; raises if not.
"""

from typing import List, Optional, Tuple

import numpy as np

from thali_pipeline.pipeline.bev import fallback_corners, warp_to_bev
from thali_pipeline.pipeline.detect_classify import ClassifierBundle, detect_and_classify
from thali_pipeline.models.sam2_segmenter import Sam2TraySegmenter
from thali_pipeline.models.yolo_detector import YoloDishDetector


def run_task2(
    img_bgr: np.ndarray,
    detector: YoloDishDetector,
    classifier_bundle: ClassifierBundle,
    classifier_mode: str,
    min_box_px: int,
    crop_padding_frac: float = 0.15,
) -> Tuple[List[dict], np.ndarray]:
    """Direct detection on the image as given. Returns (detections, annotation_base_image)."""
    detections = detect_and_classify(
        img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, crop_padding_frac
    )
    return detections, img_bgr


def run_task3(
    img_bgr: np.ndarray,
    detector: YoloDishDetector,
    classifier_bundle: ClassifierBundle,
    classifier_mode: str,
    min_box_px: int,
    segmenter: Sam2TraySegmenter,
    bev_out_size: Tuple[int, int] = (600, 450),
    crop_padding_frac: float = 0.15,
) -> Tuple[List[dict], np.ndarray]:
    """SAM2 tray segmentation -> BEV warp -> detect on the warped image.
    Returns (detections, bev_image) — detections' bboxes are in BEV-image
    coordinates, so the caller must annotate/crop against `bev_image`, NOT
    the original.
    """
    corners = segmenter.segment_tray_corners(img_bgr)
    if corners is None:
        print("[router] SAM2 found no tray contour; using fallback corners.")
        corners = fallback_corners(img_bgr)

    bev_img = warp_to_bev(img_bgr, corners, out_size=bev_out_size)
    detections = detect_and_classify(
        bev_img, detector, classifier_bundle, classifier_mode, min_box_px, crop_padding_frac
    )
    return detections, bev_img


def route_and_detect(
    img_bgr: np.ndarray,
    detector: YoloDishDetector,
    classifier_bundle: ClassifierBundle,
    classifier_mode: str,
    min_box_px: int,
    segmenter: Sam2TraySegmenter,
    mode: str = "auto",
    min_direct_detections: int = 1,
    bev_out_size: Tuple[int, int] = (600, 450),
    manual_task: Optional[int] = None,
    crop_padding_frac: float = 0.15,
) -> Tuple[List[dict], np.ndarray, int]:
    """Returns (detections, annotation_base_image_bgr, task_used)."""

    if mode == "manual":
        if manual_task not in (2, 3):
            raise ValueError("routing mode='manual' requires manual_task=2 or 3")
        if manual_task == 2:
            dets, base = run_task2(
                img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, crop_padding_frac
            )
        else:
            dets, base = run_task3(
                img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, segmenter,
                bev_out_size, crop_padding_frac,
            )
        return dets, base, manual_task

    if mode == "bev_first":
        dets, base = run_task3(
            img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, segmenter,
            bev_out_size, crop_padding_frac,
        )
        return dets, base, 3

    # auto (default)
    dets, base = run_task2(
        img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, crop_padding_frac
    )
    if len(dets) >= min_direct_detections:
        return dets, base, 2

    print(f"[router] Task 2 found {len(dets)} detection(s) (< {min_direct_detections}); "
          f"falling back to Task 3 (SAM2 -> BEV).")
    dets3, base3 = run_task3(
        img_bgr, detector, classifier_bundle, classifier_mode, min_box_px, segmenter,
        bev_out_size, crop_padding_frac,
    )
    return dets3, base3, 3
