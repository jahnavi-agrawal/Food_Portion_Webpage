from __future__ import annotations
import cv2
import numpy as np
from ..interfaces import PlateGeometryEstimator, PlateSegmenter
from ..types import PlateGeometry

class ContourPlateSegmenter(PlateSegmenter):
    """Largest plausible image contour; a baseline to replace with a learned model."""
    def __init__(self, min_area_fraction: float = 0.10) -> None: self.min_area_fraction = min_area_fraction
    def segment(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        # An image-border contour is usually the table/background, not the tray.
        # Discard it before selecting the plate contour.
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            touches_frame = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
            if cv2.contourArea(contour) >= h * w * self.min_area_fraction and not touches_frame:
                candidates.append(contour)
        mask = np.zeros((h, w), dtype=bool)
        if candidates:
            cv2.drawContours(mask.view(np.uint8), [max(candidates, key=cv2.contourArea)], -1, 1, -1)
        else:
            # Never default to the entire image: that would make table and scale
            # pixels eligible for food segmentation. This conservative fallback
            # keeps only the central field of view until a learned plate model is used.
            margin_x, margin_y = max(1, int(w * 0.05)), max(1, int(h * 0.05))
            mask[margin_y:h - margin_y, margin_x:w - margin_x] = True
        return mask

class SimplePlateGeometryEstimator(PlateGeometryEstimator):
    def estimate(self, image: np.ndarray, plate_mask: np.ndarray) -> PlateGeometry:
        # Compartments/ridges are intentionally empty until a tray adapter supplies them.
        return PlateGeometry(plate_mask=plate_mask, compartments=[], ridge_mask=np.zeros_like(plate_mask))
