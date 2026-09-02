"""Class-label loading. All classifiers and the YOLO head share the SAME
80-class ordering (configs/labels.txt) — index order must match training
exactly, so don't re-sort this file.
"""

from pathlib import Path
from typing import List


def load_labels(path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"labels file is empty: {path}")
    return labels
