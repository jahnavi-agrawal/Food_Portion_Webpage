"""SAM2-based tray registration.

The previous registration path (still kept as a fallback — see
`ColorComponentFoodSegmenter._legacy_registration` in `color_components.py`)
found the tray's four corners with a plain Canny edge + `minAreaRect` on
whatever contour `ContourPlateSegmenter` happened to find, on BOTH the
reference empty-tray image and every incoming photo. That's brittle exactly
when it matters: if the capture angle/plane changes between the reference
photo and a meal photo, `minAreaRect` fits a bounding box to a *foreshortened,
non-rectangular* silhouette, so the resulting homography is subtly wrong and
the food-floor polygons land in the wrong place on the meal photo — which is
the "doesn't pick the compartments properly" bug. Coverage computed against a
misregistered floor polygon under- or over-counts food, which is why the
current gram predictions are unreliable whenever the photo isn't captured
from the same plane as the reference image.

This module replaces that corner extraction with SAM2, which segments the
actual tray silhouette (not just an edge contour) and is far more robust to
lighting and viewing-angle changes. It's used for BOTH:
  * registering a meal photo onto the empty-tray reference frame
    (`SamTrayRegistrar.registration_homography`), and
  * the full oblique -> bird's-eye-view (BEV) warp for Task 3 style photos
    with no fixed capture angle (`SamTrayRegistrar.bev_homography`).

If SAM2 weights aren't available, every method here returns None and callers
fall back to the legacy path automatically — nothing hard-crashes when the
checkpoints haven't been downloaded yet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


class SamTrayRegistrar:
    """Segments the tray with SAM2 and extracts its 4 corners for homography."""

    def __init__(
        self,
        sam2_config: str,
        sam2_base_checkpoint: str,
        sam2_finetuned_checkpoint: str | None = None,
        device: str = "auto",
        box_prompt_margin_frac: float = 0.05,
    ) -> None:
        self.sam2_config = sam2_config
        self.sam2_base_checkpoint = sam2_base_checkpoint
        self.sam2_finetuned_checkpoint = sam2_finetuned_checkpoint
        self.box_prompt_margin_frac = box_prompt_margin_frac
        self._device_str = device
        self._predictor = None
        self._unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        if self._unavailable_reason is not None:
            return False
        if self._predictor is not None:
            return True
        return Path(self.sam2_base_checkpoint).exists()

    def _ensure_loaded(self) -> bool:
        if self._predictor is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        if not Path(self.sam2_base_checkpoint).exists():
            self._unavailable_reason = f"SAM2 checkpoint not found: {self.sam2_base_checkpoint}"
            return False
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            self._unavailable_reason = f"sam2/torch not installed ({exc})"
            return False

        device = self._device_str
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            model = build_sam2(self.sam2_config, self.sam2_base_checkpoint, device=device)
            if self.sam2_finetuned_checkpoint and Path(self.sam2_finetuned_checkpoint).exists():
                state = torch.load(self.sam2_finetuned_checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(state)
            self._predictor = SAM2ImagePredictor(model)
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._unavailable_reason = f"failed to load SAM2: {exc}"
            return False
        return True

    def tray_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Full-resolution boolean tray mask, or None if segmentation fails."""
        if not self._ensure_loaded():
            return None
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)
        mx, my = int(w * self.box_prompt_margin_frac), int(h * self.box_prompt_margin_frac)
        box = np.array([[mx, my, w - mx, h - my]])
        masks, scores, _ = self._predictor.predict(
            point_coords=None, point_labels=None, box=box, multimask_output=True,
        )
        mask = masks[int(np.argmax(scores))].astype(bool)
        return mask if mask.any() else None

    def tray_corners(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Ordered (4, 2) float32 tray corners (TL, TR, BR, BL), or None."""
        mask = self.tray_mask(image_bgr)
        if mask is None:
            return None
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        eps = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2))
        return _order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))

    def registration_homography(
        self, image_bgr: np.ndarray, reference_corners: np.ndarray
    ) -> Optional[np.ndarray]:
        """Homography mapping `image_bgr` pixel coords onto the reference-image frame."""
        current = self.tray_corners(image_bgr)
        if current is None:
            return None
        return cv2.getPerspectiveTransform(current, reference_corners.astype(np.float32))

    def bev_homography(
        self, image_bgr: np.ndarray, out_size: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Homography mapping `image_bgr` pixel coords onto a canonical BEV canvas."""
        corners = self.tray_corners(image_bgr)
        if corners is None:
            return None
        out_w, out_h = out_size
        dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
        H, _ = cv2.findHomography(corners, dst)
        return H

    @staticmethod
    def warp_to_bev(
        image_bgr: np.ndarray, homography: np.ndarray, out_size: tuple[int, int], apply_clahe: bool = True
    ) -> np.ndarray:
        out_w, out_h = out_size
        warped = cv2.warpPerspective(image_bgr, homography, (out_w, out_h))
        if not apply_clahe:
            return warped
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
