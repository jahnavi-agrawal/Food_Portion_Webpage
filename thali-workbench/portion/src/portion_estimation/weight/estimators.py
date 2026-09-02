from __future__ import annotations

import numpy as np

from ..interfaces import VolumeEstimator, WeightEstimator
from ..types import Calibration, FoodInstance, VolumeEstimate, WeightEstimate


class AreaHeightVolumeEstimator(VolumeEstimator):
    def __init__(self, depth_range_cm: float = 2.0, min_height_cm: float = 0.45, area_height_slope: float = 0.06,
                 max_height_cm: float = 2.5, liquid_fill_height_cm: float = 1.0, depth_is_metric_cm: bool = False) -> None:
        """Estimate height from depth when available, otherwise a footprint prior.

        `area_height_slope` controls the weak prior in cm / sqrt(cm^2). It prevents
        a relative-depth model from turning every food estimate into 0 g while a
        real metric depth adapter can still supply a larger height.

        `max_height_cm` is always the compartment's physical floor-to-rim depth
        (2.5 cm for the Ashoka tray). `factory.py` sources this from
        `configs/tray/ashoka.yaml` (`layout.compartment_depth_mm`) rather than a
        duplicated number here, so it can't drift out of sync with the tray file.

        `depth_is_metric_cm`: when True (set automatically by `factory.py` for
        `MetricMonoDepthEstimator`), `depth_map` already holds real centimetres
        per pixel and is used directly with no reference subtraction or
        `depth_range_cm` scaling. When False (the `LuminanceRelativeDepthEstimator`
        fallback), `depth_map` is a unitless [0, 1] relative map and is converted
        to a *weak* height signal via `depth_range_cm`, exactly as before.
        """
        self.depth_range_cm = depth_range_cm
        self.min_height_cm = min_height_cm
        self.area_height_slope = area_height_slope
        self.max_height_cm = max_height_cm
        self.liquid_fill_height_cm = min(liquid_fill_height_cm, max_height_cm)
        self.depth_is_metric_cm = depth_is_metric_cm

    def estimate(self, food: FoodInstance, depth_map: np.ndarray, calibration: Calibration,
                 plate_mask: np.ndarray) -> VolumeEstimate:
        px = int(food.mask.sum())
        area = px * calibration.cm2_per_pixel
        if self.depth_is_metric_cm:
            depth_height = float(np.mean(depth_map[food.mask])) if food.mask.any() else 0.0
            depth_height = max(0.0, min(depth_height, self.max_height_cm))
        else:
            reference = float(np.median(depth_map[plate_mask])) if plate_mask.any() else 0.0
            depth_height = max(0.0, float(np.mean(depth_map[food.mask]) - reference) * self.depth_range_cm)
        if food.physical_type == "liquid":
            # A liquid can never be taller than the compartment itself, so the
            # ceiling clamp is the SAME max_height_cm as solids, not a separate
            # number — liquid overflowing its compartment isn't physically
            # possible in this tray.
            mean_height = min(self.max_height_cm, max(depth_height, self.liquid_fill_height_cm))
            method = "compartment_area_liquid_fill_prior"
        else:
            footprint_prior = self.min_height_cm + self.area_height_slope * np.sqrt(area)
            mean_height = min(self.max_height_cm, max(depth_height, float(footprint_prior)))
            method = "solid_mask_area_height_prior"
        return VolumeEstimate(area * mean_height, area, mean_height, method)


class DensityWeightEstimator(WeightEstimator):
    def __init__(self, default_density_g_ml: float = 0.85, densities_g_ml: dict[str, float] | None = None,
                 solid_density_g_ml: float | None = None, liquid_density_g_ml: float = 1.0) -> None:
        self.default_density, self.densities = default_density_g_ml, densities_g_ml or {}
        self.type_densities = {"solid": solid_density_g_ml if solid_density_g_ml is not None else default_density_g_ml,
                               "liquid": liquid_density_g_ml}

    def estimate(self, volume: VolumeEstimate, food: FoodInstance) -> WeightEstimate:
        density = self.densities.get(food.label, self.type_densities.get(food.physical_type, self.default_density))
        return WeightEstimate(volume.volume_ml * density, .70 if food.physical_type == "liquid" else .45,
                              f"{food.physical_type}_density")
