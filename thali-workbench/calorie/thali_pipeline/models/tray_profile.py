"""Tray-aware portion scaling.

A generic thali has no absolute size reference, so portion_scaling in
calorie_attach.py falls back to "this dish's box vs. the average box in this
photo" -- relative, and it degrades to a no-op (scale=1.0) if there's only
one dish in frame.

For a tray with a MEASURED profile (real mm dimensions + per-compartment
floor polygons, e.g. configs/tray_profiles/ashoka_six_compartment_380x280.yaml),
we can do better: each compartment's real floor area is fixed geometry, known
in advance, independent of what's detected in any particular photo. A
compartment 2x the tray's average floor area gets a 2x serving-size
multiplier, always -- not just "when something big happened to be next to it
in this shot."

Matching a detection to a compartment assumes the image being scored is
framed similarly (top-down, full tray visible) to the reference empty-tray
photo the polygons were drawn on -- true for Task 3's BEV output, and for a
well-framed direct Task 2 photo.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def _polygon_area(points_norm: List[List[float]], width_mm: float, height_mm: float) -> float:
    """Shoelace formula, after scaling normalized [0,1] coords to real mm.
    Assumes the reference image's frame corresponds to the tray's full outer
    boundary (0,0)-(1,1) -> (0,0)-(width_mm,height_mm) -- the same
    "known_tray_geometry" assumption the profile's calibration.preferred_strategy
    names. A homography-based refinement (calibration.fallback_strategy) would
    tighten this further but isn't implemented here.
    """
    pts_mm = [(x * width_mm, y * height_mm) for x, y in points_norm]
    n = len(pts_mm)
    area = 0.0
    for i in range(n):
        x1, y1 = pts_mm[i]
        x2, y2 = pts_mm[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _centroid_norm(points_norm: List[List[float]]) -> Tuple[float, float]:
    xs = [p[0] for p in points_norm]
    ys = [p[1] for p in points_norm]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _label_compartments(centroids: List[Tuple[float, float]]) -> List[str]:
    """Purely geometric row/column labels ("top_left", "bottom_center", ...)
    from centroid positions -- no hardcoded knowledge of this specific tray.
    Only meaningful for a roughly 2-row layout; falls back to "slot_N" for
    anything else.
    """
    n = len(centroids)
    if n % 2 != 0 or n == 0:
        return [f"slot_{i}" for i in range(n)]

    per_row = n // 2
    indexed = sorted(range(n), key=lambda i: centroids[i][1])  # sort by y
    top_row = sorted(indexed[:per_row], key=lambda i: centroids[i][0])   # sort by x
    bottom_row = sorted(indexed[per_row:], key=lambda i: centroids[i][0])

    col_names = (
        ["left", "center", "right"] if per_row == 3
        else ["left", "right"] if per_row == 2
        else [f"col{i}" for i in range(per_row)]
    )

    labels = [""] * n
    for col, idx in enumerate(top_row):
        labels[idx] = f"top_{col_names[col]}"
    for col, idx in enumerate(bottom_row):
        labels[idx] = f"bottom_{col_names[col]}"
    return labels


class TrayProfile:
    def __init__(self, profile_id: str, compartments: List[dict]):
        """compartments: list of dicts, one per compartment:
            {"label": str, "polygon_norm": [[x,y],...], "centroid_norm": (x,y),
             "floor_area_mm2": float, "relative_size": float}
        `relative_size` = this compartment's floor area / mean floor area
        across all compartments in this profile -- fixed at load time.
        """
        self.profile_id = profile_id
        self.compartments = compartments
        self.compartment_count = len(compartments)

    def match_compartment(
        self,
        bbox: List[float],
        image_shape: Tuple[int, int],
        max_norm_distance: float = 0.35,
    ) -> Optional[int]:
        """image_shape: (height, width) of the image the bbox came from.
        Returns the index of the nearest compartment by centroid distance,
        or None if even the nearest one is further than `max_norm_distance`
        away (normalized image-diagonal-ish units) -- guards against a
        stray/false-positive detection outside the tray getting force-matched
        to whatever compartment happens to be closest.
        """
        h, w = image_shape
        cx = ((bbox[0] + bbox[2]) / 2.0) / w
        cy = ((bbox[1] + bbox[3]) / 2.0) / h

        best_idx, best_dist = None, float("inf")
        for i, comp in enumerate(self.compartments):
            comp_cx, comp_cy = comp["centroid_norm"]
            dist = ((cx - comp_cx) ** 2 + (cy - comp_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_idx, best_dist = i, dist

        if best_idx is None or best_dist > max_norm_distance:
            return None
        return best_idx

    def relative_size_for(self, compartment_idx: int) -> float:
        return self.compartments[compartment_idx]["relative_size"]

    def label_for(self, compartment_idx: int) -> str:
        return self.compartments[compartment_idx]["label"]


def load_tray_profile(path: str) -> TrayProfile:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tray profile not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    profile_id = raw["profile"]["id"]
    width_mm = raw["outer"]["width_mm"]
    height_mm = raw["outer"]["height_mm"]
    polygons = raw["food_masking"]["normalized_floor_polygons"]

    areas_mm2 = [_polygon_area(poly, width_mm, height_mm) for poly in polygons]
    centroids = [_centroid_norm(poly) for poly in polygons]
    labels = _label_compartments(centroids)
    mean_area = sum(areas_mm2) / len(areas_mm2)

    compartments = [
        {
            "label": labels[i],
            "polygon_norm": polygons[i],
            "centroid_norm": centroids[i],
            "floor_area_mm2": round(areas_mm2[i], 1),
            "relative_size": areas_mm2[i] / mean_area if mean_area > 0 else 1.0,
        }
        for i in range(len(polygons))
    ]

    return TrayProfile(profile_id, compartments)
