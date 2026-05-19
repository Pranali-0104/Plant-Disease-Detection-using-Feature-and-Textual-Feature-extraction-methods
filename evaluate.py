from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.coco import register_coco_splits
from models.detector import build_detectron_cfg
from utils.config import load_yaml, project_root
from utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Detectron2 Faster R-CNN checkpoint.")
    parser.add_argument("--config", default="configs/faster_rcnn_R50_FPN.yaml")
    parser.add_argument("--weights", default=None, help="Checkpoint path. Defaults to model_final.pth in output_dir.")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    try:
        from detectron2.data import build_detection_test_loader
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    except ImportError as exc:
        raise ImportError("Detectron2 is required for evaluation.") from exc

    args = parse_args()
    root = project_root()
    config = load_yaml(args.config)
    dataset_names = register_coco_splits(config, root)
    cfg = build_detectron_cfg(config, dataset_names, root)
    logger = setup_logger(output_dir=cfg.OUTPUT_DIR)

    weights = Path(args.weights) if args.weights else Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if not weights.is_absolute():
        weights = root / weights
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.DATASETS.TEST = (dataset_names[args.split],)

    evaluator = COCOEvaluator(dataset_names[args.split], output_dir=str(Path(cfg.OUTPUT_DIR) / f"{args.split}_evaluation"))
    data_loader = build_detection_test_loader(cfg, dataset_names[args.split])

    from detectron2.engine import DefaultTrainer

    model = DefaultTrainer.build_model(cfg)
    from detectron2.checkpoint import DetectionCheckpointer

    DetectionCheckpointer(model).load(str(weights))
    results = inference_on_dataset(model, data_loader, evaluator)
    output_path = Path(cfg.OUTPUT_DIR) / "evaluation_report.json"
    report = {
        "split": args.split,
        "weights": str(weights),
        "detectron2_metrics": results,
        "notes": [
            "COCOEvaluator reports bbox mAP for detection configs.",
            "For Mask R-CNN configs with valid COCO segmentation polygons, COCOEvaluator also reports segm mAP.",
            "Mask IoU, Dice, precision, and recall utilities are available in utils.segmentation_metrics for per-mask analysis.",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s evaluation results: %s", args.split, results)


if __name__ == "__main__":
    main()
