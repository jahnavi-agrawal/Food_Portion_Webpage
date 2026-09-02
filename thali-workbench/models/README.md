# Checkpoints

Drop these here. Every config in the project points at this folder, so nothing
else needs editing.

| File | Used by |
| --- | --- |
| `yolo_best.pt` | dish detection (both pipelines) |
| `convnext_base_ep018_macro_f1=0.9425.pt` | ConvNeXt classifier (all three tabs) |
| `save_best_dino_v2.pth` | DINOv2 classifier (calorie tab default) |
| `calorie_clip.pt` | CalorieCLIP kcal regression |
| `sam2.1_hiera_tiny.pt` | SAM2 base — tray segmentation for BEV |
| `sam2_finetuned.pth` | SAM2 fine-tuned (optional; falls back to base) |
| `labels.txt` | the 80 class names, in training order — already here |

Don't re-sort `labels.txt`: index order has to match how the classifiers were
trained.

From `calorie/`, `python scripts/check_weights.py` loads each file into its
assumed architecture and reports PASS / WARN / FAIL, which is the fastest way to
catch a checkpoint sitting in the wrong slot.
