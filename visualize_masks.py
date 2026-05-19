from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from datasets.coco import split_paths
from utils.config import load_yaml, project_root, resolve_path
from utils.mask_visualization import blend_mask, draw_combined


def polygons_to_mask(polygons: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    mask_image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_image)
    for polygon in polygons:
        if len(polygon) >= 6:
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, outline=1, fill=1)
    return np.asarray(mask_image).astype(bool)


def visualize_ground_truth_masks(config: dict, split_name: str, output_dir: Path, max_images: int) -> int:
    root = project_root()
    dataset_cfg = config["dataset"]
    split = split_paths(root / dataset_cfg["root"], split_name, dataset_cfg.get("annotation_file", "_annotations.coco.json"))
    data = json.loads(split.annotation_file.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("segmentation"):
            anns_by_image[ann["image_id"]].append(ann)

    written = 0
    for image_info in data.get("images", []):
        annotations = anns_by_image.get(image_info["id"], [])
        if not annotations:
            continue
        image_path = split.image_dir / image_info["file_name"]
        image_size = (int(image_info["width"]), int(image_info["height"]))
        combined = np.zeros((image_size[1], image_size[0]), dtype=bool)
        labels = []
        boxes = []
        for ann in annotations:
            combined |= polygons_to_mask(ann["segmentation"], image_size)
            x, y, w, h = [float(value) for value in ann["bbox"]]
            boxes.append([x, y, x + w, y + h])
            labels.append(categories.get(ann["category_id"], str(ann["category_id"])))
        label = ", ".join(sorted(set(labels)))
        out_path = output_dir / f"gt_mask_{written + 1}_{image_path.stem}.jpg"
        draw_combined(image_path, combined, boxes[0] if boxes else None, label, out_path, alpha=float(config.get("visualization", {}).get("overlay_alpha", 0.45)))
        written += 1
        if written >= max_images:
            break
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize COCO ground-truth or predicted masks.")
    parser.add_argument("--config", default="configs/mask_rcnn_R50_FPN.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--prediction-mask", default=None, help="Path to an exported transparent PNG mask.")
    parser.add_argument("--image", default=None, help="Image path for prediction mask overlay.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    root = project_root()
    output_dir = resolve_path(args.output_dir or config.get("visualization", {}).get("output_dir", "outputs/mask_visualizations"), root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ground_truth:
        split_name = config["dataset"].get(f"{args.split}_split", args.split)
        count = visualize_ground_truth_masks(
            config,
            split_name,
            output_dir / args.split / "ground_truth_masks",
            args.max_images or int(config.get("visualization", {}).get("max_images", 5)),
        )
        print(f"Wrote {count} ground-truth mask visualizations")

    if args.prediction_mask:
        if not args.image:
            raise ValueError("--image is required with --prediction-mask")
        mask = np.asarray(Image.open(args.prediction_mask).convert("RGBA"))[..., 3] > 0
        out_path = output_dir / "prediction_mask_overlay.jpg"
        blend_mask(Path(args.image), mask, out_path, alpha=float(config.get("visualization", {}).get("overlay_alpha", 0.45)))
        print(f"Wrote prediction mask overlay to {out_path}")


if __name__ == "__main__":
    main()
