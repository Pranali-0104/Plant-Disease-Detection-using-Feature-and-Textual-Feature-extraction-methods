import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a Roboflow COCO dataset with train/valid/test splits.")
    parser.add_argument("--data-dir", default="../COCO DATASET")
    parser.add_argument("--output", default="outputs/coco_dataset_audit.json")
    return parser.parse_args()


def split_report(split_dir):
    annotation_file = split_dir / "_annotations.coco.json"
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    image_ids = {image["id"] for image in data.get("images", [])}
    image_files = {image["file_name"] for image in data.get("images", [])}

    missing_images = []
    for file_name in image_files:
        if not (split_dir / file_name).exists():
            missing_images.append(file_name)

    invalid_boxes = []
    annotations_per_image = Counter()
    annotations_per_class = Counter()
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        annotations_per_image[image_id] += 1
        annotations_per_class[categories.get(category_id, str(category_id))] += 1

        if image_id not in image_ids:
            invalid_boxes.append({"id": annotation.get("id"), "reason": "missing image_id"})
            continue
        if not bbox or len(bbox) != 4:
            invalid_boxes.append({"id": annotation.get("id"), "reason": "missing bbox"})
            continue
        _x, _y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            invalid_boxes.append({"id": annotation.get("id"), "reason": "non-positive bbox"})

    images_without_annotations = sorted(image_ids - set(annotations_per_image.keys()))

    return {
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(categories),
        "missing_images": missing_images[:25],
        "missing_images_count": len(missing_images),
        "invalid_boxes": invalid_boxes[:25],
        "invalid_boxes_count": len(invalid_boxes),
        "images_without_annotations_count": len(images_without_annotations),
        "annotations_per_class": dict(sorted(annotations_per_class.items())),
    }


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_path = (script_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {}
    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            report[split] = {"error": f"Missing split: {split_dir}"}
            continue
        report[split] = split_report(split_dir)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset: {data_dir}")
    for split, details in report.items():
        if "error" in details:
            print(f"{split}: {details['error']}")
            continue
        print(
            f"{split}: images={details['images']} annotations={details['annotations']} "
            f"categories={details['categories']} missing_images={details['missing_images_count']} "
            f"invalid_boxes={details['invalid_boxes_count']} "
            f"images_without_annotations={details['images_without_annotations_count']}"
        )
    print(f"Audit written to {output_path}")


if __name__ == "__main__":
    main()
