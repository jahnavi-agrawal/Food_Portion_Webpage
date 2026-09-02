"""SAM2 tray segmentation -> 4-corner extraction, for Task 3's BEV step.

Ported near-verbatim from miscellaneous/predict_thali.py (`_segment_tray`,
`_order_corners`), since that logic is already working in your repo. Lazily
loads SAM2 only when a Task-3 path is actually taken, so Task-2-only runs
never need the `sam2` package installed.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return rect


class Sam2TraySegmenter:
    def __init__(
        self,
        base_checkpoint: str,
        finetuned_checkpoint: Optional[str],
        model_cfg: str,
        device: torch.device,
    ):
        self.base_checkpoint = base_checkpoint
        self.finetuned_checkpoint = finetuned_checkpoint
        self.model_cfg = model_cfg
        self.device = device
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return

        base_ckpt = Path(self.base_checkpoint)
        if not base_ckpt.exists():
            raise FileNotFoundError(f"SAM2 base checkpoint not found: {base_ckpt}")

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(self.model_cfg, str(base_ckpt), device=self.device)

        finetuned = Path(self.finetuned_checkpoint) if self.finetuned_checkpoint else None
        if finetuned and finetuned.exists():
            state = torch.load(finetuned, map_location=self.device, weights_only=False)
            sam2_model.load_state_dict(state)
            print(f"[sam2] fine-tuned weights loaded: {finetuned}")
        else:
            print("[sam2] using base SAM2 weights (no fine-tuned checkpoint found)")

        self._predictor = SAM2ImagePredictor(sam2_model)

    def segment_tray_corners(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Returns ordered (4, 2) float32 corner points, or None on failure."""
        self._ensure_loaded()

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(img_rgb)

        # Loose bounding box covering ~90% of the image as the SAM2 prompt.
        mx, my = int(w * 0.05), int(h * 0.05)
        box = np.array([[mx, my, w - mx, h - my]])

        masks, scores, _ = self._predictor.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=True,
        )

        mask = masks[np.argmax(scores)].astype(np.uint8)
        if mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2).astype(np.float32))
        box_pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        return order_corners(box_pts)
