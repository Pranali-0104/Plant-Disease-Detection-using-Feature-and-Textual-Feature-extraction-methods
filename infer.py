from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from datasets.coco import class_names, register_coco_splits, split_paths
from inference.predictor import run_inference
from models.detector import build_detectron_cfg
from utils.config import load_yaml, project_root, resolve_path
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Faster R-CNN inference on one image or a folder.")
    parser.add_argument("--config", default="configs/faster_rcnn_R50_FPN.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(args.config)
    dataset_names = register_coco_splits(config, root)
    cfg = build_detectron_cfg(config, dataset_names, root)

    infer_cfg = config.get("inference", {})
    input_path = resolve_path(args.input or infer_cfg.get("input", "COCO DATASET/test"), root)
    output_dir = resolve_path(args.output_dir or infer_cfg.get("output_dir", "outputs/inference"), root)
    weights = Path(args.weights) if args.weights else Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    cfg.MODEL.WEIGHTS = str(weights)
    score_threshold = args.score_threshold if args.score_threshold is not None else float(infer_cfg.get("score_threshold", 0.5))
    top_k = args.top_k if args.top_k is not None else int(infer_cfg.get("top_k", 1))
    extensions = infer_cfg.get("extensions", [".jpg", ".jpeg", ".png"])

    logger = setup_logger(output_dir=output_dir)
    dataset_cfg = config["dataset"]
    train_split = split_paths(root / dataset_cfg["root"], dataset_cfg.get("train_split", "train"), dataset_cfg.get("annotation_file", "_annotations.coco.json"))
    names = class_names(train_split.annotation_file)

    results = run_inference(
        cfg=cfg,
        input_path=input_path,
        output_dir=output_dir,
        class_names=names,
        score_threshold=score_threshold,
        top_k=top_k,
        extensions=extensions,
        app_config=config,
    )
    (output_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_severity_csv(results, output_dir / "severity_results.csv")
    logger.info("Inference complete for %s images. Outputs: %s", len(results), output_dir)


def write_severity_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = [
        "image",
        "disease_name",
        "confidence",
        "severity_percent",
        "severity_class",
        "bbox",
        "mask_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            detections = result.get("detections") or []
            if not detections:
                writer.writerow({"image": result.get("image")})
            for detection in detections:
                writer.writerow(
                    {
                        "image": result.get("image"),
                        "disease_name": detection.get("disease_name"),
                        "confidence": detection.get("confidence"),
                        "severity_percent": detection.get("severity_percent"),
                        "severity_class": detection.get("severity_class"),
                        "bbox": json.dumps(detection.get("bbox")),
                        "mask_path": detection.get("mask_path"),
                    }
                )


if __name__ == "__main__":
    main()
