# Images

Put meal photos here for batch runs (`python scripts/run_batch.py --images images/`).

`empty_tray.jpg` is special: a top-down photo of the **empty** Ashoka tray with
the complete outer rim visible. The food segmenter differences each meal photo
against it, which is much more accurate than saturation thresholding alone, and
`normalized_floor_polygons` in `configs/tray/ashoka.yaml` were drawn on it (at
1061 × 673). Batch runs skip it automatically — it's calibration data, not a
meal.

Without it everything still runs, just with the weaker saturation-only masks;
the app warns when it's missing.
