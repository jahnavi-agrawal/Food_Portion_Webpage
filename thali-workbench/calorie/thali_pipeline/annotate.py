"""Draw boxes + "class (calories kcal)" labels + a summary banner onto the
image. Uses cv2 (as the rest of the repo already does) rather than PIL, so no
extra font-file dependency.
"""

from typing import List

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)          # green, BGR
TEXT_COLOR = (0, 0, 0)           # black text on a light background chip
LABEL_BG_COLOR = (0, 255, 0)
BANNER_BG_COLOR = (30, 30, 30)
BANNER_TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_label_chip(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.55):
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    y_top = max(0, y - th - baseline - 4)
    cv2.rectangle(img, (x, y_top), (x + tw + 6, y_top + th + baseline + 4), LABEL_BG_COLOR, -1)
    cv2.putText(img, text, (x + 3, y_top + th + 1), FONT, scale, TEXT_COLOR, 1, cv2.LINE_AA)


def annotate_image(
    base_image_bgr: np.ndarray,
    detections: List[dict],
    task_used: int,
    total_calories_summed: float,
    classifier_mode: str = "dino",
) -> np.ndarray:
    img = base_image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)

        cal = det.get("calories")
        cal_text = f"{cal:.0f} kcal" if cal is not None else "n/a"
        label = f"{det['class']} ({det['class_confidence']:.2f}) | {cal_text}"
        _draw_label_chip(img, label, x1, y1)

    banner_lines = [
        f"Task {task_used} | classifier: {classifier_mode} | dishes: {len(detections)}",
        f"Sum of per-dish estimates: {total_calories_summed:.0f} kcal",
    ]

    h, w = img.shape[:2]
    banner_h = 24 * len(banner_lines) + 12
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), BANNER_BG_COLOR, -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    for i, line in enumerate(banner_lines):
        cv2.putText(img, line, (8, 20 + i * 24), FONT, 0.55, BANNER_TEXT_COLOR, 1, cv2.LINE_AA)

    return img
