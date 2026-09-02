from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .types import Calibration, FoodInstance, PlateGeometry, VolumeEstimate, WeightEstimate

class PlateSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray: ...

class PlateGeometryEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, plate_mask: np.ndarray) -> PlateGeometry: ...

class FoodSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, plate_mask: np.ndarray,
                calibration: Calibration | None = None) -> list[FoodInstance]: ...

class FoodClassifier(ABC):
    @abstractmethod
    def classify(self, image: np.ndarray, foods: list[FoodInstance]) -> list[FoodInstance]: ...

class DepthEstimator(ABC):
    @abstractmethod
    def estimate(self, image: np.ndarray, plate_mask: np.ndarray,
                  visible_floor_mask: np.ndarray | None = None) -> np.ndarray: ...

class Calibrator(ABC):
    @abstractmethod
    def calibrate(self, geometry: PlateGeometry, image_shape: tuple[int, int]) -> Calibration: ...

class VolumeEstimator(ABC):
    @abstractmethod
    def estimate(self, food: FoodInstance, depth_map: np.ndarray, calibration: Calibration, plate_mask: np.ndarray) -> VolumeEstimate: ...

class WeightEstimator(ABC):
    @abstractmethod
    def estimate(self, volume: VolumeEstimate, food: FoodInstance) -> WeightEstimate: ...
