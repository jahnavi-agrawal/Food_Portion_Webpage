from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

Mask = np.ndarray

@dataclass(frozen=True)
class PlateGeometry:
    plate_mask: Mask
    compartments: list[Mask] = field(default_factory=list)
    ridge_mask: Mask | None = None

@dataclass(frozen=True)
class FoodInstance:
    mask: Mask
    score: float = 1.0
    label: str = "unknown"
    class_confidence: float = 0.0
    physical_type: str = "solid"
    compartment_coverage: float = 0.0

@dataclass(frozen=True)
class Calibration:
    mm_per_pixel_x: float
    mm_per_pixel_y: float
    confidence: float
    method: str
    homography: np.ndarray | None = None

    @property
    def cm2_per_pixel(self) -> float:
        return self.mm_per_pixel_x * self.mm_per_pixel_y / 100.0

@dataclass(frozen=True)
class VolumeEstimate:
    volume_ml: float
    area_cm2: float
    mean_height_cm: float
    method: str

@dataclass(frozen=True)
class WeightEstimate:
    grams: float
    confidence: float
    method: str

@dataclass(frozen=True)
class FoodResult:
    food: FoodInstance
    volume: VolumeEstimate
    weight: WeightEstimate

@dataclass(frozen=True)
class PipelineResult:
    geometry: PlateGeometry
    visible_plate_mask: Mask
    calibration: Calibration
    depth_map: np.ndarray
    foods: list[FoodResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"calibration": {"mm_per_pixel_x": self.calibration.mm_per_pixel_x, "mm_per_pixel_y": self.calibration.mm_per_pixel_y, "confidence": self.calibration.confidence, "method": self.calibration.method}, "foods": [{"label": f.food.label, "physical_type": f.food.physical_type, "compartment_coverage": f.food.compartment_coverage, "classification_confidence": f.food.class_confidence, "volume_ml": f.volume.volume_ml, "area_cm2": f.volume.area_cm2, "mean_height_cm": f.volume.mean_height_cm, "weight_g": f.weight.grams, "weight_confidence": f.weight.confidence} for f in self.foods], "total_weight_g": sum(f.weight.grams for f in self.foods), "metadata": self.metadata}
