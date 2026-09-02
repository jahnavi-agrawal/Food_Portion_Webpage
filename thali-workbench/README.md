# Thali Workbench

Dish classification, calorie estimation and Ashoka-tray portion estimation, in
one Streamlit app over the two existing pipelines.

```
thali-workbench/
  app/          Streamlit app — three tabs over both pipelines
  calorie/      detection -> classification -> CalorieCLIP, with BEV fallback
  portion/      tray calibration -> depth -> volume -> grams
  models/       every checkpoint lives here (not in git)
```

Both pipelines stay independent packages with their own CLIs. The app imports
them as libraries and copies none of their model code, so a fix in either one
shows up in the app immediately.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"   # BEV / Task 3
```

Then drop your checkpoints into `models/` (see `models/README.md` for the exact
filenames) and put a top-down photo of the empty Ashoka tray at
`portion/images/empty_tray.jpg`.

```bash
streamlit run app/DishDetect.py
```

Everything resolves relative to this folder, so no environment variables are
needed for the default layout. The sidebar lists anything it can't find.

## The three tabs

**Dish detection** — YOLO finds every dish in the photo, ConvNeXt names each one.
A close-up with no detections falls back to classifying the whole frame.

**Calorie estimation (general case)** — YOLO finds the dishes, the chosen
classifier (DINOv2, ConvNeXt or both) names each crop, CalorieCLIP regresses
kcal per crop, and each estimate is pulled toward the class lookup in
proportion to classifier confidence. No tray profile is used here, so portion
size comes from each box's area relative to the other dishes in the same photo.
Routing is selectable: Auto runs direct detection and falls back to BEV
(SAM2 tray corners → homography → CLAHE) when it finds nothing, or force either
path from the dropdown.

**Ashoka thali portion** — the measured tray profile supplies the scale, so no
marker or reference object is needed. Compartment-floor polygons find the food,
Depth Anything V2 gives per-pixel heights self-calibrated against the exposed
floor and clamped by the tray's real 2.5 cm depth, then volume × density gives
grams.

### BEV in the portion tab

For angled photos, SAM2 segments the tray and the app warps it onto a canonical
top-down canvas before the pipeline runs. The tray is placed inside a 40 px
blank margin rather than filling the canvas: `ContourPlateSegmenter` discards
any contour touching the frame (it assumes that's the table), so the margin is
what makes the plate mask come out as exactly the tray. That makes calibration
exact — 380 mm over a known 600 px — instead of depending on camera distance.
Expect roughly **0.633 mm/px** in the readout when BEV is on; it's a quick way
to tell whether registration worked. Turn BEV off for square-on photos.

## Command line

Both pipelines still run standalone. Run each from inside its own folder, since
their configs use relative paths.

```bash
cd calorie
python scripts/check_weights.py                      # verify every checkpoint loads
python scripts/predict_image.py plate.jpg --classifier dino
python scripts/predict_image.py plate.jpg --task 3   # force BEV
python scripts/predict_folder.py images/ --outdir outputs/batch1
python scripts/debug_yolo.py plate.jpg               # when detection returns nothing

cd portion
pip install -e .
portion-estimate --image plate.jpg --config configs/baseline.yaml --visualization out.png
python scripts/run_batch.py --images images/ --config configs/baseline.yaml
```

`configs/baseline.yaml` is the Ashoka tray; `configs/general.yaml` swaps the
floor-polygon segmenter for direct YOLO detection when there's no known
compartment layout.

## Things worth knowing

**`calorie_lookup.csv` is simulated.** The per-100g and serving-size figures in
`calorie/configs/calorie_lookup.csv` cover all 80 classes so the blending path
works end to end, but they're plausible estimates, not USDA or IFCT figures.
Swap in real values before trusting any absolute number. Lower
`calorie_lookup.max_weight` in `weights.yaml` to reduce how much they pull the
CalorieCLIP output, or set `enabled: false` to ignore them entirely.

**One number, one place.** The tray's 2.5 cm floor-to-rim depth lives only in
`portion/configs/tray/ashoka.yaml` (`layout.compartment_depth_mm`). `factory.py`
reads it from there to clamp both the volume estimator and the depth model, so
don't reintroduce a `max_height_cm` in `baseline.yaml`.

**Classifier preprocessing.** The original `DishDetect.py` used
`Resize((200, 200))` with no normalization, `weights.yaml` specifies resize 438
/ crop 384 with ImageNet normalization, and the portion repo's classifier used
224. Those can't all be right for one checkpoint. The app standardizes on the
`weights.yaml` values, since that's the documented training config — change
`RESIZE` / `CROP_SIZE` in `app/models.py` to go back.

**Checkpoint shape.** `DishDetect.py` treated the ConvNeXt file as a pickled
`nn.Module`; both pipelines treat it as a `state_dict`.
`app/models.load_dish_classifier` accepts either, and the portion pipeline gets
its dish names through `app/models.DishNameClassifier` so both tabs share one
loaded model instead of two.

**First run needs network.** DINOv2 comes from `torch.hub` and Depth Anything V2
from HuggingFace; both are cached locally afterwards.
