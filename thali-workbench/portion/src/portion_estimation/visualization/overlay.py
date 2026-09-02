from __future__ import annotations
import cv2
import numpy as np
from ..types import PipelineResult

def render_overlay(image: np.ndarray, result: PipelineResult, show_visible_plate: bool = False,
                    min_label_weight_g: float = 3.0, show_class_label: bool = True) -> np.ndarray:
    """Render food instances by default; plate geometry is opt-in for debugging.

    Each food gets its mask tinted plus a two-line label: dish name (or
    "solid"/"liquid" if no classifier was attached) on top, grams below.
    """
    out = image.copy()
    if show_visible_plate:
        tint = np.zeros_like(image); tint[result.visible_plate_mask] = (255, 180, 0)
        out = cv2.addWeighted(out, 1.0, tint, 0.18, 0)
    for index, item in enumerate(result.foods, 1):
        color = tuple(int(v) for v in np.random.default_rng(index).integers(60, 255, 3))
        out[item.food.mask] = (0.45 * out[item.food.mask] + 0.55 * np.array(color)).astype(np.uint8)
        ys, xs = np.where(item.food.mask)
        if not len(xs) or item.weight.grams < min_label_weight_g:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        weight_text = f'{item.weight.grams:.1f}g'
        if show_class_label and item.food.label not in ('solid', 'liquid'):
            label_text = item.food.label
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv2.putText(out, label_text, (cx - lw // 2, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
            (ww, wh), _ = cv2.getTextSize(weight_text, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv2.putText(out, weight_text, (cx - ww // 2, cy + wh + 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        else:
            (ww, wh), _ = cv2.getTextSize(weight_text, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv2.putText(out, weight_text, (cx - ww // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
    return out
