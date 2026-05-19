import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, category_id_to_label):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.category_id_to_label = category_id_to_label

        data = json.loads(self.annotation_file.read_text(encoding="utf-8"))
        self.categories = {category["id"]: category["name"] for category in data["categories"]}
        self.images = data["images"]

        annotations_by_image = defaultdict(list)
        for annotation in data["annotations"]:
            bbox = annotation.get("bbox")
            if bbox and len(bbox) == 4 and float(bbox[2]) > 0 and float(bbox[3]) > 0:
                annotations_by_image[annotation["image_id"]].append(annotation)
        self.annotations_by_image = annotations_by_image

        self.images = [
            image
            for image in self.images
            if annotations_by_image.get(image["id"]) and (self.image_dir / image["file_name"]).exists()
        ]
        if not self.images:
            raise RuntimeError(f"No annotated images found for {self.annotation_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = self.images[index]
        image_path = self.image_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        for annotation in self.annotations_by_image[image_info["id"]]:
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            boxes.append([x, y, x + width, y + height])
            labels.append(self.category_id_to_label[annotation["category_id"]])
            areas.append(width * height)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return F.to_tensor(image), target, image_path


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_categories(annotation_file):
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda category: category["id"])
    category_id_to_label = {category["id"]: index + 1 for index, category in enumerate(categories)}
    label_to_name = {index + 1: category["name"] for index, category in enumerate(categories)}
    return category_id_to_label, label_to_name


def build_model(num_classes, pretrained):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def limit_dataset(dataset, max_images):
    if max_images is None or max_images >= len(dataset):
        return dataset
    return Subset(dataset, range(max_images))


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def average_precision(tp_flags, fp_flags, total_gt):
    if total_gt == 0:
        return None
    if not tp_flags:
        return 0.0

    tp = np.cumsum(np.array(tp_flags, dtype=np.float32))
    fp = np.cumsum(np.array(fp_flags, dtype=np.float32))
    recall = tp / max(float(total_gt), 1.0)
    precision = tp / np.maximum(tp + fp, 1e-8)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])

    changing_points = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changing_points + 1] - recall[changing_points]) * precision[changing_points + 1]))


def filtered_prediction_items(prediction, score_threshold, max_predictions):
    items = []
    boxes = prediction["boxes"].detach().cpu().tolist()
    labels = prediction["labels"].detach().cpu().tolist()
    scores = prediction["scores"].detach().cpu().tolist()

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue
        items.append((box, int(label), float(score)))
        if max_predictions is not None and len(items) >= max_predictions:
            break
    return items


@torch.no_grad()
def evaluate_ap50(model, data_loader, device, score_threshold, max_predictions):
    model.eval()
    gt_counts = defaultdict(int)
    predictions_by_class = defaultdict(list)

    for image_index, (images, targets, _paths) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        predictions = model(images)

        for batch_index, prediction in enumerate(predictions):
            target = targets[batch_index]
            gt_boxes = target["boxes"].detach().cpu().tolist()
            gt_labels = target["labels"].detach().cpu().tolist()
            matched = [False] * len(gt_boxes)

            for label in gt_labels:
                gt_counts[int(label)] += 1

            for pred_box, pred_label, pred_score in filtered_prediction_items(
                prediction,
                score_threshold,
                max_predictions,
            ):
                best_iou = 0.0
                best_gt_index = None
                for gt_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if matched[gt_index] or int(gt_label) != pred_label:
                        continue
                    iou = box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_index = gt_index

                is_tp = best_gt_index is not None and best_iou >= 0.5
                if is_tp:
                    matched[best_gt_index] = True
                predictions_by_class[pred_label].append(
                    {
                        "score": pred_score,
                        "tp": 1 if is_tp else 0,
                        "fp": 0 if is_tp else 1,
                    }
                )

    ap_by_class = {}
    for label, total_gt in gt_counts.items():
        class_predictions = sorted(
            predictions_by_class.get(label, []),
            key=lambda item: item["score"],
            reverse=True,
        )
        ap = average_precision(
            [item["tp"] for item in class_predictions],
            [item["fp"] for item in class_predictions],
            total_gt,
        )
        ap_by_class[label] = 0.0 if ap is None else ap

    mean_ap50 = float(np.mean(list(ap_by_class.values()))) if ap_by_class else 0.0
    return {"mAP50": mean_ap50, "AP50_by_label": ap_by_class, "gt_counts": dict(gt_counts)}


@torch.no_grad()
def validation_loss(model, data_loader, device):
    was_training = model.training
    model.train()
    total = 0.0
    count = 0
    for images, targets, _paths in data_loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        total += sum(loss.item() for loss in losses.values())
        count += 1
    model.train(was_training)
    return total / max(count, 1)


def draw_predictions(image_path, prediction, label_to_name, output_path, score_threshold, max_predictions):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    kept = 0

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()

    drawn = 0
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue
        if max_predictions is not None and drawn >= max_predictions:
            break
        x1, y1, x2, y2 = [float(value) for value in box]
        name = label_to_name.get(int(label), str(int(label)))
        text = f"{name} {float(score):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        text_box = draw.textbbox((x1, y1), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 4)
        draw.rectangle((x1, label_y, x1 + text_width + 6, label_y + text_height + 4), fill="red")
        draw.text((x1 + 3, label_y + 2), text, fill="white", font=font)
        kept += 1
        drawn += 1

    image.save(output_path, quality=95)
    return kept


def parse_args():
    parser = argparse.ArgumentParser(description="Train or smoke-test Faster R-CNN on Roboflow COCO data.")
    parser.add_argument("--data-dir", default="../COCO DATASET")
    parser.add_argument("--output-dir", default="outputs/fasterrcnn_coco")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-valid-images", type=int, default=None)
    parser.add_argument("--max-test-images", type=int, default=5)
    parser.add_argument("--max-predictions", type=int, default=1, help="Maximum boxes to draw per test image.")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--pretrained", action="store_true", help="Use COCO pretrained weights if available.")
    parser.add_argument("--dry-run", action="store_true", help="Load data/model and exit before training.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    prediction_dir = output_dir / "test_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    train_ann = data_dir / "train" / "_annotations.coco.json"
    valid_ann = data_dir / "valid" / "_annotations.coco.json"
    test_ann = data_dir / "test" / "_annotations.coco.json"

    category_id_to_label, label_to_name = load_categories(train_ann)
    (output_dir / "labels.json").write_text(json.dumps(label_to_name, indent=2), encoding="utf-8")

    train_dataset = limit_dataset(
        CocoDetectionDataset(data_dir / "train", train_ann, category_id_to_label),
        args.max_train_images,
    )
    valid_dataset = limit_dataset(
        CocoDetectionDataset(data_dir / "valid", valid_ann, category_id_to_label),
        args.max_valid_images,
    )
    test_dataset = limit_dataset(
        CocoDetectionDataset(data_dir / "test", test_ann, category_id_to_label),
        args.max_test_images,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(label_to_name) + 1, pretrained=args.pretrained).to(device)

    print(f"Device: {device}")
    print(f"Classes: {len(label_to_name)}")
    print(f"Train images used: {len(train_dataset)}")
    print(f"Valid images used: {len(valid_dataset)}")
    print(f"Test images used: {len(test_dataset)}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        print("Dry run complete.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    metrics_path = output_dir / "metrics.csv"
    best_valid_loss = float("inf")
    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=["epoch", "train_loss", "valid_loss", "lr"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for step, (images, targets, _paths) in enumerate(train_loader, start=1):
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                losses = model(images, targets)
                loss = sum(loss_value for loss_value in losses.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item()
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

            train_loss = total / max(len(train_loader), 1)
            valid_loss = validation_loss(model, valid_loader, device)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "valid_loss": f"{valid_loss:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )
            metrics_file.flush()
            print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

            checkpoint = {"model": model.state_dict(), "labels": label_to_name, "epoch": epoch}
            torch.save(checkpoint, output_dir / "last_checkpoint.pth")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(checkpoint, output_dir / "best_checkpoint.pth")
                print(f"Saved best checkpoint with valid_loss={valid_loss:.4f}")
            lr_scheduler.step()

    torch.save(
        {"model": model.state_dict(), "labels": label_to_name},
        output_dir / "fasterrcnn_coco_smoke_test.pth",
    )

    model.roi_heads.score_thresh = args.score_threshold
    eval_metrics = evaluate_ap50(model, test_loader, device, args.score_threshold, args.max_predictions)
    metrics_json = {
        "score_threshold": args.score_threshold,
        "max_predictions": args.max_predictions,
        "mAP50": eval_metrics["mAP50"],
        "AP50_by_class": {
            label_to_name.get(label, str(label)): ap for label, ap in eval_metrics["AP50_by_label"].items()
        },
        "gt_counts": {
            label_to_name.get(label, str(label)): count for label, count in eval_metrics["gt_counts"].items()
        },
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print(f"Test mAP@0.50={eval_metrics['mAP50']:.4f}")

    model.eval()
    prediction_counts = []
    with torch.no_grad():
        for index, (images, _targets, paths) in enumerate(test_loader, start=1):
            images = [image.to(device) for image in images]
            predictions = model(images)
            output_path = prediction_dir / f"prediction_{index}_{paths[0].stem}.jpg"
            kept = draw_predictions(
                paths[0],
                predictions[0],
                label_to_name,
                output_path,
                args.score_threshold,
                args.max_predictions,
            )
            prediction_counts.append(kept)
            print(f"Prediction {index}: boxes>={args.score_threshold} = {kept} | {output_path}")

    print(f"Prediction previews saved: {len(prediction_counts)}")
    print(f"Images with at least one predicted box: {sum(1 for count in prediction_counts if count > 0)}")


if __name__ == "__main__":
    main()
