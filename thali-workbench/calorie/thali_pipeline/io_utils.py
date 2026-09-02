"""Small IO helpers: image loading, JSON/CSV writers."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image (unsupported format or bad path): {path}")
    return img


def save_image_bgr(path: str, img_bgr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img_bgr)
    if not ok:
        raise IOError(f"failed to write image: {path}")


def save_json(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv_rows(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
