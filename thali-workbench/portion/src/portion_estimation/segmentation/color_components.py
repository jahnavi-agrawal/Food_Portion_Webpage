from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import yaml

from ..interfaces import FoodSegmenter
from ..types import Calibration, FoodInstance
from .sam_tray import SamTrayRegistrar


def _ordered_corners(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    ordered = np.empty((4, 2), np.float32)
    sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
    return ordered


def _mask_corners(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return _ordered_corners(cv2.boxPoints(cv2.minAreaRect(contour))) if cv2.contourArea(contour) >= 100 else None


class ColorComponentFoodSegmenter(FoodSegmenter):
    """Segment food in empty-tray reference coordinates, then map masks back.

    Registration (mapping a photo onto the reference empty-tray frame so the
    known compartment-floor polygons line up) is tried in this order:

      1. SAM2 tray segmentation (`sam_registrar`, see `sam_tray.py`) — robust
         to the photo being taken from a different plane/angle than the
         reference photo, which was the source of misaligned compartments
         and bad coverage in the old minAreaRect-only path.
      2. Legacy minAreaRect homography (`_legacy_registration`) — kept as an
         automatic fallback so the pipeline still runs with no SAM2
         checkpoints downloaded, just less robustly to plane changes.
      3. No registration — falls back to running the floor polygons directly
         on the raw image coordinates (only correct if the photo already
         matches the reference framing closely).
    """

    def __init__(self, min_area_px: int = 500, saturation_threshold: int = 55,
                 boundary_margin_px: int = 12, tray_profile_path: str | None = None,
                 reference_difference_threshold: int = 18, solid_min_coverage: float = .025,
                 use_sam_registration: bool = True, sam_registrar: SamTrayRegistrar | None = None) -> None:
        self.min_area_px, self.saturation_threshold = min_area_px, saturation_threshold
        self.boundary_margin_px = boundary_margin_px
        self.reference_difference_threshold, self.solid_min_coverage = reference_difference_threshold, solid_min_coverage
        self.profile: dict[str, object] = {}
        self.reference_image: np.ndarray | None = None
        self.reference_floors: list[np.ndarray] = []
        self.reference_corners: np.ndarray | None = None
        self.use_sam_registration = use_sam_registration
        self.sam_registrar = sam_registrar
        self.mask_dilation_cm: float = 0.0
        if tray_profile_path:
            with Path(tray_profile_path).open(encoding="utf-8") as handle:
                self.profile = yaml.safe_load(handle) or {}
            path = self.profile.get("food_masking", {}).get("reference_image_path")
            if path:
                self.reference_image = cv2.imread(str(path))
                if self.reference_image is None:
                    raise FileNotFoundError(f"Cannot read empty tray reference: {path}")
                self.reference_floors = self._compartments(self.reference_image.shape[:2])
            self.mask_dilation_cm = float(self.profile.get("food_masking", {}).get("mask_dilation_cm", 0.0))

    def _compartments(self, shape: tuple[int, int]) -> list[np.ndarray]:
        height, width = shape
        masks: list[np.ndarray] = []
        for polygon in self.profile.get("food_masking", {}).get("normalized_floor_polygons", []):
            points = np.array([(round(x * width), round(y * height)) for x, y in polygon], np.int32)
            mask = np.zeros((height, width), np.uint8)
            cv2.fillPoly(mask, [points], 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.boundary_margin_px + 1,) * 2)
            masks.append(cv2.erode(mask, kernel).astype(bool))
        return masks

    def _legacy_registration(self, plate_mask: np.ndarray) -> np.ndarray | None:
        """Original single minAreaRect-based homography. See class docstring."""
        if self.reference_image is None:
            return None
        current = _mask_corners(plate_mask)
        if self.reference_corners is None:
            gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.morphologyEx(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = gray.shape
            choices = []
            for contour in contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                image_frame = x <= 1 and y <= 1 and x + cw >= w - 1 and y + ch >= h - 1
                if cv2.contourArea(contour) > .10 * h * w and not image_frame:
                    choices.append(contour)
            if choices:
                mask = np.zeros((h, w), np.uint8)
                cv2.drawContours(mask, [max(choices, key=cv2.contourArea)], -1, 1, -1)
                self.reference_corners = _mask_corners(mask)
        return cv2.getPerspectiveTransform(current, self.reference_corners) if current is not None and self.reference_corners is not None else None

    def _reference_corners_from_sam(self) -> np.ndarray | None:
        """Corners of the reference (empty-tray) image, extracted with SAM2 once and cached."""
        if self.reference_image is None or self.sam_registrar is None:
            return None
        if not hasattr(self, "_sam_reference_corners"):
            self._sam_reference_corners = self.sam_registrar.tray_corners(self.reference_image)
        return self._sam_reference_corners

    def _registration(self, image: np.ndarray, plate_mask: np.ndarray) -> np.ndarray | None:
        if self.use_sam_registration and self.sam_registrar is not None and self.sam_registrar.available:
            ref_corners = self._reference_corners_from_sam()
            if ref_corners is not None:
                homography = self.sam_registrar.registration_homography(image, ref_corners)
                if homography is not None:
                    return homography
        return self._legacy_registration(plate_mask)

    @staticmethod
    def _reference_difference(image: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
        lab, ref = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.int16), cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.int16)
        if valid.any():
            lab[..., 0] = np.clip(lab[..., 0] - int(np.median(lab[..., 0][valid]) - np.median(ref[..., 0][valid])), 0, 255)
        delta = np.abs(lab - ref)
        return np.maximum(delta[..., 0], np.maximum(delta[..., 1], delta[..., 2])).astype(np.uint8)

    def _dilation_kernel(self, calibration: Calibration | None) -> np.ndarray | None:
        if self.mask_dilation_cm <= 0:
            return None
        if calibration is None:
            return None
        mm_per_px = (calibration.mm_per_pixel_x + calibration.mm_per_pixel_y) / 2.0
        if mm_per_px <= 0:
            return None
        radius_px = max(1, round(self.mask_dilation_cm * 10.0 / mm_per_px))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))

    def segment(self, image: np.ndarray, plate_mask: np.ndarray,
                calibration: Calibration | None = None) -> list[FoodInstance]:
        homography = self._registration(image, plate_mask)
        if self.reference_image is not None and homography is not None and self.reference_floors:
            normalized = cv2.warpPerspective(image, homography, (self.reference_image.shape[1], self.reference_image.shape[0]))
            floors, reference, inverse = self.reference_floors, self.reference_image, np.linalg.inv(homography)
        else:
            normalized, floors, reference, inverse = image, (self._compartments(image.shape[:2]) or [plate_mask.astype(bool)]), None, None
        hsv = cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
        if reference is None:
            changed = hsv[..., 1] >= self.saturation_threshold
        else:
            changed = self._reference_difference(normalized, reference, np.logical_or.reduce(floors)) >= self.reference_difference_threshold
        candidate = np.logical_or(changed, hsv[..., 1] >= self.saturation_threshold).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        liquid = self.profile.get("food_masking", {}).get("liquid", {})
        min_coverage, max_gray_std = float(liquid.get("min_floor_coverage", .72)), float(liquid.get("max_gray_std", 42))
        gray, foods = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY), []
        dilation_kernel = self._dilation_kernel(calibration)
        for floor in floors:
            actual = candidate.astype(bool) & floor
            coverage = float(actual.sum()) / max(1, int(floor.sum()))
            if coverage < self.solid_min_coverage:
                continue
            is_liquid = coverage >= min_coverage and float(np.std(gray[actual])) <= max_gray_std
            mask = floor if is_liquid else actual
            # Grow the mask outward by `mask_dilation_cm` (default 1 cm) before
            # mapping back to image coordinates. Compensates for masks that
            # undershoot the true food footprint — thin sauce/gravy rims and
            # low-contrast edges that fall just under `reference_difference_threshold`
            # — which was under-counting coverage and therefore under-predicting
            # grams. Liquids already use the full floor polygon so dilation is a
            # no-op for them; it only affects the tighter solid masks.
            if dilation_kernel is not None and not is_liquid:
                mask = cv2.dilate(mask.astype(np.uint8), dilation_kernel).astype(bool) & floor
            if inverse is not None:
                mask = cv2.warpPerspective(mask.astype(np.uint8), inverse, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST).astype(bool)
            if int(mask.sum()) >= self.min_area_px:
                foods.append(FoodInstance(mask=mask, score=coverage, label="liquid" if is_liquid else "solid",
                                          class_confidence=.65 if reference is not None else .35,
                                          physical_type="liquid" if is_liquid else "solid", compartment_coverage=coverage))
        return foods
