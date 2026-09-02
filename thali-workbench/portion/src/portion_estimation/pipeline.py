from __future__ import annotations
import logging
import numpy as np
from .types import FoodResult, PipelineResult

LOGGER = logging.getLogger(__name__)

class PortionEstimationPipeline:
    def __init__(self, **components: object) -> None:
        self.__dict__.update(components)
    def predict(self, image: np.ndarray) -> PipelineResult:
        if image is None or image.ndim != 3: raise ValueError('Expected a BGR RGB-image array')
        plate = self.plate_segmenter.segment(image)
        geometry = self.geometry_estimator.estimate(image, plate)
        # Calibration is computed before food segmentation (not after, as
        # before) so the food segmenter can convert a physical mask-dilation
        # margin (cm) into the right number of pixels for THIS image, rather
        # than a fixed pixel count that's wrong whenever zoom/distance differs.
        calibration = self.calibrator.calibrate(geometry, image.shape[:2])
        foods = self.food_segmenter.segment(image, plate, calibration)
        # `ufunc.reduce(initial=...)` accepts a scalar identity, not an image-shaped
        # mask. An explicit union also handles the no-food case predictably.
        occupied = np.zeros_like(plate, dtype=bool)
        for food in foods:
            occupied |= food.mask
        visible = plate & ~occupied
        depth = self.depth_estimator.estimate(image, plate, visible)
        if getattr(self, 'food_classifier', None) is not None:
            foods = self.food_classifier.classify(image, foods)
        results = []
        for food in foods:
            volume = self.volume_estimator.estimate(food, depth, calibration, plate)
            results.append(FoodResult(food, volume, self.weight_estimator.estimate(volume, food)))
        LOGGER.info('Estimated %d food instances', len(results))
        return PipelineResult(geometry, visible, calibration, depth, results, {'num_foods': len(results)})
