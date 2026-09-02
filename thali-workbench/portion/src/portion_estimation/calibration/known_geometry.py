from __future__ import annotations
import cv2
from ..interfaces import Calibrator
from ..types import Calibration, PlateGeometry

class KnownGeometryCalibrator(Calibrator):
    """Scale from a configured physical plate/tray envelope, never from hard-coded dimensions."""
    def __init__(self, reference_width_mm: float, reference_height_mm: float) -> None:
        if min(reference_width_mm, reference_height_mm) <= 0: raise ValueError("Reference dimensions must be positive")
        self.width_mm, self.height_mm = reference_width_mm, reference_height_mm
    def calibrate(self, geometry: PlateGeometry, image_shape: tuple[int, int]) -> Calibration:
        mask = geometry.plate_mask.astype('uint8')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: raise ValueError("Cannot calibrate without a plate contour")
        _, _, width_px, height_px = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if width_px < 2 or height_px < 2: raise ValueError("Plate contour is too small")
        return Calibration(self.width_mm / width_px, self.height_mm / height_px, 0.75, "known_geometry")
