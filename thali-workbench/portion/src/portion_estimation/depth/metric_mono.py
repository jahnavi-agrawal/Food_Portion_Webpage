"""Calibrated monocular depth using a frozen Depth Anything V2 checkpoint.

Why not fine-tune Depth Anything on your own images
-----------------------------------------------------
Fine-tuning a depth network needs paired (RGB, metric-depth) ground truth —
e.g. from a depth camera, stereo rig, or a ruler measurement per pixel/region.
A dish-classification dataset (N images per food class) has no depth channel
at all, so there is nothing to fine-tune *against* even with unlimited
images of that kind. With only ~9 images/class it would also be far too
little data to fine-tune a ViT-based depth backbone without destroying its
pretrained priors.

What this does instead: zero-shot inference + per-image self-calibration
--------------------------------------------------------------------------
Depth Anything V2 is used frozen, exactly as released. Its raw output is a
*relative* inverse-depth map — correctly ordered (closer = higher value) but
with no metric scale and an arbitrary per-image offset. Two things anchor it
to real centimetres on every single image, with no training data required:

  1. **Zero anchor** — the compartment floor pixels NOT covered by any food
     (`visible_floor_mask`, i.e. `plate_mask & ~occupied`) are, by definition,
     sitting at the bottom of the compartment. Their median relative-depth
     value is defined as height = 0 cm.
  2. **Ceiling clamp** — nothing can physically sit higher than the
     compartment wall. `compartment_depth_cm` (2.5 cm for the Ashoka tray,
     read from `configs/tray/ashoka.yaml` — see `factory.py`) hard-clamps
     the top of the scale.

The relative values inside food regions are linearly rescaled between those
two anchors. This is a *calibration*, not a model fit — it re-derives the
correct scale/shift fresh for every photo, so it isn't thrown off by
lighting or exposure differences between photos the way the old luminance
placeholder was.

Optional: `scripts/calibrate_depth_scale.py` lets you tighten the mapping
later using a handful (10-30) of ruler-measured examples, if you ever want
to collect them — that is a 2-parameter regression, not a network fine-tune,
so it doesn't need per-class data either.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from ..interfaces import DepthEstimator

LOGGER = logging.getLogger(__name__)


class MetricMonoDepthEstimator(DepthEstimator):
    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        device: str = "auto",
        compartment_depth_cm: float = 2.5,
        min_floor_reference_px: int = 200,
        fallback: DepthEstimator | None = None,
        manual_scale: float = 1.0,
        manual_shift: float = 0.0,
    ) -> None:
        self.model_id = model_id
        self._device_str = device
        self.compartment_depth_cm = compartment_depth_cm
        self.min_floor_reference_px = min_floor_reference_px
        self.fallback = fallback
        # Optional fine correction from scripts/calibrate_depth_scale.py, applied
        # AFTER the built-in floor/ceiling self-calibration below. Defaults to a
        # no-op (1.0, 0.0) — most users never need to touch this.
        self.manual_scale = manual_scale
        self.manual_shift = manual_shift
        self._pipe = None
        self._unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        if self._unavailable_reason is not None:
            return False
        return True  # unknown until first real load attempt

    def _ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            self._unavailable_reason = f"torch/transformers not installed ({exc})"
            LOGGER.warning(
                "MetricMonoDepthEstimator: %s — falling back to luminance depth.",
                self._unavailable_reason,
            )
            return False

        device = self._device_str
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            device = 0
        elif device == "cpu":
            device = -1

        try:
            self._pipe = pipeline(task="depth-estimation", model=self.model_id, device=device)
        except Exception as exc:  # pragma: no cover - network/environment dependent
            self._unavailable_reason = f"failed to load {self.model_id} ({exc})"
            LOGGER.warning(
                "MetricMonoDepthEstimator: %s — falling back to luminance depth.",
                self._unavailable_reason,
            )
            return False
        return True

    def _raw_relative_depth(self, image_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        result = self._pipe(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)
        if depth.shape[:2] != image_bgr.shape[:2]:
            depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return depth

    def estimate(self, image: np.ndarray, plate_mask: np.ndarray,
                 visible_floor_mask: np.ndarray | None = None) -> np.ndarray:
        if not self._ensure_loaded():
            if self.fallback is not None:
                return self.fallback.estimate(image, plate_mask, visible_floor_mask)
            return np.zeros(image.shape[:2], dtype=np.float32)

        raw = self._raw_relative_depth(image)

        floor_ref = visible_floor_mask if visible_floor_mask is not None else plate_mask
        if floor_ref is None or int(floor_ref.sum()) < self.min_floor_reference_px:
            LOGGER.info(
                "MetricMonoDepthEstimator: too little exposed floor (%s px) to "
                "self-calibrate this image confidently; falling back.",
                0 if floor_ref is None else int(floor_ref.sum()),
            )
            if self.fallback is not None:
                return self.fallback.estimate(image, plate_mask, visible_floor_mask)
            return np.zeros_like(raw)

        # Anchor 1: exposed floor -> 0 cm (robust to outliers via median).
        floor_value = float(np.median(raw[floor_ref]))

        # Depth Anything's convention can be "larger = closer" or
        # "larger = farther" depending on checkpoint; resolve the sign using
        # the plate itself, since food (closer to camera in a top-down or
        # near-top-down shot) should read as HIGHER height than the floor
        # once oriented correctly.
        plate_values = raw[plate_mask] if plate_mask.any() else raw.ravel()
        food_like = raw > np.percentile(plate_values, 75)
        sign = 1.0 if np.median(raw[food_like & plate_mask]) >= floor_value else -1.0 if food_like.any() else 1.0

        relative_height = sign * (raw - floor_value)
        relative_height = np.clip(relative_height, 0.0, None)

        # Anchor 2: clamp to the physical ceiling. Rescale by the 95th
        # percentile *within the plate* (not the raw max) so a couple of
        # noisy specular-highlight pixels don't compress everything else
        # toward zero.
        plate_heights = relative_height[plate_mask] if plate_mask.any() else relative_height.ravel()
        if plate_heights.size == 0 or float(np.percentile(plate_heights, 95)) <= 1e-6:
            if self.fallback is not None:
                return self.fallback.estimate(image, plate_mask, visible_floor_mask)
            return np.zeros_like(raw)
        scale_ref = float(np.percentile(plate_heights, 95))
        height_cm = np.clip(relative_height / scale_ref * self.compartment_depth_cm, 0.0, self.compartment_depth_cm)
        if self.manual_scale != 1.0 or self.manual_shift != 0.0:
            height_cm = np.clip(height_cm * self.manual_scale + self.manual_shift, 0.0, self.compartment_depth_cm)

        # `AreaHeightVolumeEstimator` expects a depth map plus a "reference"
        # baseline it subtracts itself (see weight/estimators.py) and then
        # multiplies by `depth_range_cm`. To keep that estimator's contract
        # unchanged for the luminance fallback while giving it a value that's
        # already in real centimetres here, we encode height_cm directly and
        # rely on `depth_range_cm=1.0` / reference=0 when this estimator is
        # selected (see factory.py, which sets depth_range_cm=1.0 for the
        # depth_anything pipeline since this map is pre-scaled to cm already).
        return height_cm.astype(np.float32)
