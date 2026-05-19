import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Draw COCO annotation boxes for a few images.")
    parser.add_argument("--coco-json", default="../COCO DATASET/train/_annotations.coco.json")
    parser.add_argument("--image-dir", default="../COCO DATASET/train")
    parser.add_argument("--output-dir", default="outputs/coco_box_previews")
    parser.add_argument("--max-images", type=int, default=5)
    return parser.parse_args()


def as_float(value):
    return float(value)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    coco_json = (script_dir / args.coco_json).resolve()
    image_dir = (script_dir / args.image_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}

    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if bbox and len(bbox) == 4:
            anns_by_image[ann["image_id"]].append(ann)

    previewed = 0
    summary = []
    font = ImageFont.load_default()

    for image_info in data.get("images", []):
        annotations = anns_by_image.get(image_info["id"], [])
        if not annotations:
            continue

        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            summary.append((image_info["file_name"], 0, "missing image"))
            continue

        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for ann in annotations:
                x, y, width, height = [as_float(v) for v in ann["bbox"]]
                x2 = x + width
                y2 = y + height
                label = categories.get(ann.get("category_id"), str(ann.get("category_id")))

                draw.rectangle((x, y, x2, y2), outline="red", width=3)
                text_box = draw.textbbox((x, y), label, font=font)
                text_width = text_box[2] - text_box[0]
                text_height = text_box[3] - text_box[1]
                label_y = max(0, y - text_height - 4)
                draw.rectangle((x, label_y, x + text_width + 6, label_y + text_height + 4), fill="red")
                draw.text((x + 3, label_y + 2), label, fill="white", font=font)

            out_path = output_dir / f"preview_{previewed + 1}_{Path(image_info['file_name']).stem}.jpg"
            image.save(out_path, quality=95)

        summary.append((image_info["file_name"], len(annotations), str(out_path)))
        previewed += 1
        if previewed >= args.max_images:
            break

    print(f"COCO file: {coco_json}")
    print(f"Images in JSON: {len(data.get('images', []))}")
    print(f"Annotations in JSON: {len(data.get('annotations', []))}")
    print(f"Categories: {len(categories)}")
    print(f"Preview images written: {previewed}")
    for file_name, box_count, output in summary[: args.max_images]:
        print(f"{file_name} | boxes={box_count} | {output}")

    if previewed == 0:
        raise RuntimeError("No annotated images were found to preview.")


if __name__ == "__main__":
    main()
