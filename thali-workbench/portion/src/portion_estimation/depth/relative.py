from __future__ import annotations
import cv2
import numpy as np
from ..interfaces import DepthEstimator

class LuminanceRelativeDepthEstimator(DepthEstimator):
    """Deterministic relative-depth placeholder — zero dependencies, always available.

    Used automatically as a fallback when `MetricMonoDepthEstimator` (Depth
    Anything V2) can't load (no torch/transformers, or no GPU/internet to
    fetch weights). Output is [0, 1] relative depth only, no metric meaning;
    `AreaHeightVolumeEstimator` treats it as a weak prior, not ground truth.
    """
    def estimate(self, image: np.ndarray, plate_mask: np.ndarray,
                  visible_floor_mask: np.ndarray | None = None) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Smooth inverse luminance supplies a stable non-metric height prior for baseline tests.
        depth = cv2.GaussianBlur(1.0 - gray, (0, 0), 2.0)
        valid = depth[plate_mask]
        if valid.size == 0: return np.zeros_like(depth)
        lo, hi = np.percentile(valid, [2, 98])
        return np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
