from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def draw_boxes(
    image_path: Path,
    boxes: list[list[float]],
    labels: list[str],
    output_path: Path,
    color: str = "red",
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _font()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [float(value) for value in box]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def visualize_ground_truth(
    annotation_file: Path,
    image_dir: Path,
    output_dir: Path,
    max_images: int = 5,
) -> list[Path]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        anns_by_image[annotation["image_id"]].append(annotation)

    written: list[Path] = []
    for image_info in data.get("images", []):
        annotations = anns_by_image.get(image_info["id"], [])
        if not annotations:
            continue
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            continue

        boxes = []
        labels = []
        for annotation in annotations:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            boxes.append([x, y, x + width, y + height])
            labels.append(categories.get(annotation["category_id"], str(annotation["category_id"])))

        output_path = output_dir / f"gt_{len(written) + 1}_{image_path.stem}.jpg"
        draw_boxes(image_path, boxes, labels, output_path)
        written.append(output_path)
        if len(written) >= max_images:
            break
    return written


def plot_class_distribution(annotation_file: Path, output_path: Path) -> None:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {category["id"]: category["name"] for category in data.get("categories", [])}
    counts = Counter(categories.get(ann["category_id"], str(ann["category_id"])) for ann in data.get("annotations", []))

    names = list(counts.keys())
    values = [counts[name] for name in names]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, max(6, len(names) * 0.35)))
    plt.barh(names, values, color="#2e7d32")
    plt.xlabel("Annotations")
    plt.ylabel("Class")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_training_losses(metrics_csv: Path, output_path: Path) -> None:
    epochs = []
    train_losses = []
    valid_losses = []

    with metrics_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            valid_losses.append(float(row["valid_loss"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train loss")
    plt.plot(epochs, valid_losses, marker="o", label="Valid loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
