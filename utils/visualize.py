from __future__ import annotations

import argparse
from pathlib import Path

from datasets.coco import split_paths
from utils.config import load_yaml, project_root, resolve_path
from utils.visualization import plot_class_distribution, plot_training_losses, visualize_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization utilities for the plant disease COCO dataset.")
    parser.add_argument("--config", default="configs/faster_rcnn_R50_FPN.yaml")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--ground-truth", action="store_true")
    parser.add_argument("--class-distribution", action="store_true")
    parser.add_argument("--loss-curve", action="store_true")
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(args.config)
    dataset_cfg = config["dataset"]
    vis_cfg = config.get("visualization", {})
    output_dir = resolve_path(vis_cfg.get("output_dir", "outputs/visualizations"), root)
    split_name = dataset_cfg.get(f"{args.split}_split", args.split)
    split = split_paths(root / dataset_cfg["root"], split_name, dataset_cfg.get("annotation_file", "_annotations.coco.json"))

    if args.ground_truth:
        written = visualize_ground_truth(
            split.annotation_file,
            split.image_dir,
            output_dir / args.split / "ground_truth",
            max_images=args.max_images or int(vis_cfg.get("max_images", 5)),
        )
        print(f"Wrote {len(written)} ground-truth visualizations to {output_dir / args.split / 'ground_truth'}")

    if args.class_distribution:
        path = output_dir / args.split / "class_distribution.png"
        plot_class_distribution(split.annotation_file, path)
        print(f"Wrote class distribution plot to {path}")

    if args.loss_curve:
        metrics_csv = resolve_path(args.metrics_csv or "outputs/faster_rcnn_R50_FPN/metrics.csv", root)
        path = output_dir / "training_loss.png"
        plot_training_losses(metrics_csv, path)
        print(f"Wrote training loss plot to {path}")


if __name__ == "__main__":
    main()
