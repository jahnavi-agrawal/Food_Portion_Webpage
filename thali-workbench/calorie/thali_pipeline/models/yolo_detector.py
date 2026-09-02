"""YOLO dish/thali-region detector.

Filtering approach mirrors miscellaneous/predict_thali.py: a single
YOLO.predict() call with agnostic NMS baked in via conf/iou thresholds, rather
than a separate torchvision NMS + top-k pass (the alternative style used in
the notebooks). `max_detections` below is just a safety cap on top of that.
"""

from pathlib import Path
from typing import List, Optional, TypedDict

import numpy as np


class Detection(TypedDict):
    bbox: List[float]     # [x1, y1, x2, y2] in the coordinate space of the image passed in
    det_conf: float


class YoloDishDetector:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.10,
        iou_threshold: float = 0.30,
        agnostic_nms: bool = True,
        max_detections: Optional[int] = 8,
    ):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {weights_path}. This is the one "
                f"checkpoint missing from your screenshot — set `yolo.weights` "
                f"in configs/weights.yaml to your trained detector (called "
                f"`best.pt` in the original repo's scripts)."
            )
        from ultralytics import YOLO  # deferred: heavy import, only needed here

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        self.max_detections = max_detections

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=img_bgr,
            save=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )

        boxes = results[0].boxes
        detections: List[Detection] = []
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().item())
            detections.append({"bbox": [x1, y1, x2, y2], "det_conf": conf})

        # Highest confidence first, then cap.
        detections.sort(key=lambda d: d["det_conf"], reverse=True)
        if self.max_detections is not None:
            detections = detections[: self.max_detections]

        return detections
