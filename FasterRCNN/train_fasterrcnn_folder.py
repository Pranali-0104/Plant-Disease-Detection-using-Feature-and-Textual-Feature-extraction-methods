import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FolderDetectionDataset(Dataset):
    """Treat each class-folder image as one object covering the full image."""

    def __init__(self, root, class_to_idx):
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.samples = []

        for class_name in sorted(class_to_idx):
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_tensor = F.to_tensor(image)

        target = {
            "boxes": torch.tensor([[0.0, 0.0, float(width), float(height)]], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor([float(width * height)], dtype=torch.float32),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def discover_classes(*roots):
    class_names = set()
    for root in roots:
        root = Path(root)
        if root.is_dir():
            class_names.update(path.name for path in root.iterdir() if path.is_dir())
    return {class_name: index + 1 for index, class_name in enumerate(sorted(class_names))}


def build_model(num_classes, pretrained):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


@torch.no_grad()
def estimate_validation_loss(model, data_loader, device):
    was_training = model.training
    model.train()
    total_loss = 0.0
    batches = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        losses = model(images, targets)
        total_loss += sum(loss.item() for loss in losses.values())
        batches += 1
    model.train(was_training)
    return total_loss / max(batches, 1)


def maybe_limit(dataset, max_images):
    if max_images is None or max_images >= len(dataset):
        return dataset
    return Subset(dataset, range(max_images))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN on a class-folder plant disease dataset."
    )
    parser.add_argument("--data-dir", default="../dataset", help="Dataset folder containing train/valid.")
    parser.add_argument("--output-dir", default="outputs/fasterrcnn_folder", help="Where checkpoints are saved.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--max-valid-images", type=int, default=None, help="Optional quick-test limit.")
    parser.add_argument("--pretrained", action="store_true", help="Download/use COCO pretrained weights.")
    parser.add_argument("--dry-run", action="store_true", help="Check dataset/model setup and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    output_dir = (script_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = discover_classes(train_dir, valid_dir)
    if not class_to_idx:
        raise RuntimeError(f"No class folders found under {data_dir}")

    train_dataset = maybe_limit(FolderDetectionDataset(train_dir, class_to_idx), args.max_train_images)
    valid_dataset = maybe_limit(FolderDetectionDataset(valid_dir, class_to_idx), args.max_valid_images)

    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}
    (output_dir / "classes.json").write_text(json.dumps(idx_to_class, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_to_idx) + 1, pretrained=args.pretrained).to(device)

    print(f"Device: {device}")
    print(f"Classes: {len(class_to_idx)}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Valid images: {len(valid_dataset)}")
    print(f"Output: {output_dir}")

    if args.dry_run:
        print("Dry run complete. Start training by removing --dry-run.")
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

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            losses = model(images, targets)
            loss = sum(loss_value for loss_value in losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 10 == 0:
                print(f"Epoch {epoch} step {step}/{len(train_loader)} loss {running_loss / step:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        valid_loss = estimate_validation_loss(model, valid_loader, device)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        checkpoint_path = output_dir / f"fasterrcnn_epoch_{epoch}.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "classes": idx_to_class,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            },
            checkpoint_path,
        )
        print(f"Saved {checkpoint_path}")

    torch.save(model.state_dict(), output_dir / "fasterrcnn_final.pth")
    print(f"Final model saved to {output_dir / 'fasterrcnn_final.pth'}")


if __name__ == "__main__":
    main()
