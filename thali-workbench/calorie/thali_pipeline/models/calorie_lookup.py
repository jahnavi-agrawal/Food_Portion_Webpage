"""Per-class calorie lookup: converts a class label into an absolute
calorie estimate, using an assumed typical serving size (grams). This is a
SIMULATED placeholder table (configs/calorie_lookup.csv) -- swap in real
per-100g figures (USDA/IFCT/etc.) and real serving-size assumptions for
production use.
"""

import csv
from pathlib import Path
from typing import Dict, Optional


class CalorieLookupTable:
    def __init__(self, rows: Dict[str, dict]):
        self._rows = rows  # class -> {"calories_per_100g": float, "typical_serving_g": float}

    def estimate(self, label: str, portion_scale: float = 1.0) -> Optional[float]:
        """Absolute kcal estimate for `label`, or None if the class isn't in
        the table. `portion_scale` multiplies the table's typical_serving_g
        -- e.g. 1.5 for a dish whose detected box is 50% larger than average
        in this thali, 0.5 for one that's half the average size.
        """
        row = self._rows.get(label)
        if row is None:
            return None
        scaled_serving_g = row["typical_serving_g"] * portion_scale
        return row["calories_per_100g"] * scaled_serving_g / 100.0

    def has(self, label: str) -> bool:
        return label in self._rows


def load_calorie_lookup(path: str) -> CalorieLookupTable:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calorie lookup table not found: {path}")

    rows: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["class"].strip()
            rows[label] = {
                "calories_per_100g": float(row["calories_per_100g"]),
                "typical_serving_g": float(row["typical_serving_g"]),
            }

    if not rows:
        raise ValueError(f"calorie lookup table is empty: {path}")

    return CalorieLookupTable(rows)
