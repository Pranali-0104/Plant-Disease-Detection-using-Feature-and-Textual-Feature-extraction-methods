from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from datasets.coco import split_paths
from utils.config import load_yaml, project_root


def _valid_polygon(segmentation: Any) -> bool:
    if not isinstance(segmentation, list) or not segmentation:
        return False
    for polygon in segmentation:
        if not isinstance(polygon, list):
            return False
        if len(polygon) < 6 or len(polygon) % 2 != 0:
            return False
    return True


def audit_mask_split(split_dir: Path, annotation_name: str, verify_images: bool = False) -> dict[str, Any]:
    split = split_paths(split_dir.parent, split_dir.name, annotation_name)
    data = json.loads(split.annotation_file.read_text(encoding="utf-8"))
    image_lookup = {image["id"]: image for image in data.get("images", [])}

    missing_masks = []
    invalid_polygons = []
    empty_annotations = []
    corrupted_images = []

    annotations_by_image: dict[int, int] = {}
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        annotations_by_image[image_id] = annotations_by_image.get(image_id, 0) + 1
        segmentation = annotation.get("segmentation")
        if not segmentation:
            missing_masks.append(annotation.get("id"))
        elif not _valid_polygon(segmentation):
            invalid_polygons.append(annotation.get("id"))

    for image_id, image_info in image_lookup.items():
        if annotations_by_image.get(image_id, 0) == 0:
            empty_annotations.append(image_info.get("file_name"))
        if verify_images:
            image_path = split.image_dir / image_info.get("file_name", "")
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception:
                corrupted_images.append(str(image_path))

    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "missing_masks_count": len(missing_masks),
        "missing_masks_sample": missing_masks[:25],
        "invalid_polygons_count": len(invalid_polygons),
        "invalid_polygons_sample": invalid_polygons[:25],
        "empty_annotations_count": len(empty_annotations),
        "empty_annotations_sample": empty_annotations[:25],
        "corrupted_images_count": len(corrupted_images),
        "corrupted_images_sample": corrupted_images[:25],
        "corrupted_image_check": "enabled" if verify_images else "skipped",
        "mask_ready": len(missing_masks) == 0 and len(invalid_polygons) == 0 and len(corrupted_images) == 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit COCO polygon masks for Mask R-CNN readiness.")
    parser.add_argument("--config", default="configs/mask_rcnn_R50_FPN.yaml")
    parser.add_argument("--output", default="outputs/mask_dataset_audit.json")
    parser.add_argument("--verify-images", action="store_true", help="Open every image to check corruption. Slower on OneDrive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(args.config)
    dataset_cfg = config["dataset"]
    dataset_root = root / dataset_cfg["root"]
    annotation_name = dataset_cfg.get("annotation_file", "_annotations.coco.json")

    report = {}
    for key in ["train", "valid", "test"]:
        split_name = dataset_cfg.get(f"{key}_split", key)
        report[key] = audit_mask_split(dataset_root / split_name, annotation_name, verify_images=args.verify_images)

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for split, details in report.items():
        print(
            f"{split}: images={details['images']} annotations={details['annotations']} "
            f"missing_masks={details['missing_masks_count']} invalid_polygons={details['invalid_polygons_count']} "
            f"corrupted_images={details['corrupted_images_count']} "
            f"corrupted_check={details['corrupted_image_check']} mask_ready={details['mask_ready']}"
        )
    print(f"Mask audit written to {output_path}")


if __name__ == "__main__":
    main()
