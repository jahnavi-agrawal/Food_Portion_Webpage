#!/usr/bin/env python3
"""Sanity-check every checkpoint configured in configs/weights.yaml.

Run this FIRST, before any real inference — the checkpoint <-> role mapping
in weights.yaml is a best guess based on filenames alone. This script loads
each file into its assumed architecture and reports:

    PASS  -- loaded with no key mismatches
    WARN  -- loaded, but with missing/unexpected keys (wrong file for this
              role, or a genuinely different checkpoint format)
    FAIL  -- file missing, or couldn't be loaded at all

Usage:
    python scripts/check_weights.py
    python scripts/check_weights.py --config configs/weights.yaml
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thali_pipeline.config import load_config
from thali_pipeline.device import resolve_device
from thali_pipeline.labels import load_labels


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        status, detail = fn()
    except Exception as e:
        status, detail = FAIL, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    print(f"[{status}] {name}: {detail}")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weights.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    labels = load_labels(config["labels_path"])
    num_classes = len(labels)
    print(f"device: {device} | num_classes: {num_classes}")

    results = {}

    def check_convnext(profile_name):
        from thali_pipeline.models.classifiers import build_convnext_classifier
        profile_cfg = config["classifiers"][profile_name]
        p = Path(profile_cfg["weights"])
        if not p.exists():
            return FAIL, f"file not found: {p}"
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_convnext_classifier(profile_cfg, num_classes, device)
        out = buf.getvalue()
        print(out, end="")
        return (WARN if "WARNING" in out else PASS), str(p)

    def check_dino(profile_name):
        from thali_pipeline.models.classifiers import build_dino_classifier
        profile_cfg = config["classifiers"][profile_name]
        p = Path(profile_cfg["weights"])
        if not p.exists():
            return FAIL, f"file not found: {p}"
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            build_dino_classifier(profile_cfg, num_classes, device)
        out = buf.getvalue()
        print(out, end="")
        return (WARN if "WARNING" in out else PASS), str(p)

    def check_yolo():
        from thali_pipeline.models.yolo_detector import YoloDishDetector
        yolo_cfg = config["yolo"]
        p = Path(yolo_cfg["weights"])
        if not p.exists():
            return FAIL, f"file not found: {p} (this is the checkpoint missing from your screenshot)"
        YoloDishDetector(weights_path=yolo_cfg["weights"])
        return PASS, str(p)

    def check_sam2():
        from thali_pipeline.models.sam2_segmenter import Sam2TraySegmenter
        sam2_cfg = config["sam2"]
        base = Path(sam2_cfg["base_checkpoint"])
        if not base.exists():
            return FAIL, f"base checkpoint not found: {base}"
        seg = Sam2TraySegmenter(
            base_checkpoint=sam2_cfg["base_checkpoint"],
            finetuned_checkpoint=sam2_cfg.get("finetuned_checkpoint"),
            model_cfg=sam2_cfg["model_cfg"],
            device=device,
        )
        seg._ensure_loaded()  # forces the actual load, including fine-tuned weights if present
        return PASS, f"{base} (+ finetuned if present)"

    def check_calorie_clip():
        from thali_pipeline.models.calorie_clip import CalorieCLIPModel
        cc_cfg = config["calorie_clip"]
        p = Path(cc_cfg["weights"])
        if not p.exists():
            return FAIL, f"file not found: {p}"
        CalorieCLIPModel(
            weights_path=cc_cfg["weights"],
            base_model=cc_cfg.get("base_model", "ViT-B-32"),
            pretrained=cc_cfg.get("pretrained", "openai"),
            device=device,
        )
        return PASS, str(p)

    def check_calorie_lookup():
        from thali_pipeline.models.calorie_lookup import load_calorie_lookup
        lookup_cfg = config.get("calorie_lookup", {})
        if not lookup_cfg.get("enabled", True):
            return PASS, "disabled in config"
        p = Path(lookup_cfg["path"])
        if not p.exists():
            return FAIL, f"file not found: {p}"
        table = load_calorie_lookup(lookup_cfg["path"])
        missing = [lbl for lbl in labels if not table.has(lbl)]
        if missing:
            return WARN, f"{len(missing)}/{len(labels)} labels missing from lookup table: {missing[:5]}..."
        return PASS, f"{p} ({len(labels)} labels covered)"

    results["yolo"] = check("YOLO detector", check_yolo)
    results["classifier:dino"] = check("Classifier: dino", lambda: check_dino("dino"))
    results["classifier:convnext"] = check("Classifier: convnext", lambda: check_convnext("convnext"))
    results["sam2"] = check("SAM2 segmenter", check_sam2)
    results["calorie_clip"] = check("CalorieCLIP", check_calorie_clip)
    results["calorie_lookup"] = check("Calorie lookup table", check_calorie_lookup)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  [{status}] {name}")

    if any(s == FAIL for s in results.values()):
        print("\nOne or more checkpoints failed to load — fix paths in "
              "configs/weights.yaml before running real inference.")
        sys.exit(1)
    if any(s == WARN for s in results.values()):
        print("\nAll files loaded, but some had key mismatches (see WARNING "
              "lines above) — likely the wrong file assigned to that role.")
        sys.exit(2)
    print("\nAll checkpoints loaded cleanly.")


if __name__ == "__main__":
    main()
