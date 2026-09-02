from __future__ import annotations
import argparse, csv, json, logging
from pathlib import Path
import cv2, yaml
from .factory import build_components
from .pipeline import PortionEstimationPipeline
from .visualization import render_overlay

def main() -> None:
    parser = argparse.ArgumentParser(description='Estimate food weights from one RGB image.')
    parser.add_argument('--image', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--output', default='result.json'); parser.add_argument('--visualization')
    parser.add_argument('--csv-output', help='Optional CSV with one row per detected food item.')
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    image = cv2.imread(args.image)
    if image is None: raise SystemExit(f'Unable to read image: {args.image}')
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    result = PortionEstimationPipeline(**build_components(config)).predict(image)
    data = result.to_dict()
    Path(args.output).write_text(json.dumps(data, indent=2), encoding='utf-8')
    if args.visualization:
        min_label_weight_g = float(config.get('output', {}).get('min_label_weight_g', 3.0))
        cv2.imwrite(args.visualization, render_overlay(image, result, min_label_weight_g=min_label_weight_g))
    if args.csv_output and data['foods']:
        fields = list(data['foods'][0].keys())
        with Path(args.csv_output).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data['foods'])

if __name__ == '__main__': main()
