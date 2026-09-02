"""Bird's-eye-view reconstruction (Task 3, steps 3-4): homography warp from
4 tray corners + CLAHE contrast enhancement. Ported from
miscellaneous/predict_thali.py's `_reconstruct_bev`.
"""

from typing import Tuple

import cv2
import numpy as np


def warp_to_bev(
    img_bgr: np.ndarray,
    src_corners: np.ndarray,
    out_size: Tuple[int, int] = (600, 450),
    apply_clahe: bool = True,
) -> np.ndarray:
    out_w, out_h = out_size
    dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])

    H, _ = cv2.findHomography(src_corners, dst)
    warped = cv2.warpPerspective(img_bgr, H, (out_w, out_h))

    if not apply_clahe:
        return warped

    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fallback_corners(img_bgr: np.ndarray) -> np.ndarray:
    """Used only if SAM2 fails to find a 4-corner tray contour — a loose
    trapezoid guess so the pipeline degrades gracefully instead of crashing.
    Matches the fallback in miscellaneous/predict_thali.py's `predict_task3`.
    """
    h, w = img_bgr.shape[:2]
    return np.float32([
        [50, 200], [w - 50, 200],
        [w, h], [0, h],
    ])
