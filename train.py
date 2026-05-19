from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets.coco import class_names, register_coco_splits, split_paths
from dataset_audit_masks import audit_mask_split
from models.detector import build_detectron_cfg
from models.trainer import build_trainer
from utils.config import load_yaml, project_root
from utils.logging import setup_logger
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Detectron2 Faster R-CNN for plant disease detection.")
    parser.add_argument("--config", default="configs/faster_rcnn_R50_FPN.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = load_yaml(args.config)
    seed_everything(int(config.get("project", {}).get("seed", 42)))

    enforce_mask_readiness(config, root)
    dataset_names = register_coco_splits(config, root)
    cfg = build_detectron_cfg(config, dataset_names, root)
    logger = setup_logger(output_dir=cfg.OUTPUT_DIR)
    logger.info("Registered datasets: %s", dataset_names)
    logger.info("Output directory: %s", cfg.OUTPUT_DIR)

    dataset_cfg = config["dataset"]
    train_split = split_paths(root / dataset_cfg["root"], dataset_cfg.get("train_split", "train"), dataset_cfg.get("annotation_file", "_annotations.coco.json"))
    names = class_names(train_split.annotation_file)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (Path(cfg.OUTPUT_DIR) / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    with (Path(cfg.OUTPUT_DIR) / "detectron_config.yaml").open("w", encoding="utf-8") as handle:
        handle.write(cfg.dump())

    trainer = build_trainer(cfg, config)
    resume = args.resume or bool(config.get("training", {}).get("resume", True))
    trainer.resume_or_load(resume=resume)
    trainer.train()


def enforce_mask_readiness(config: dict, root: Path) -> None:
    if str(config.get("model", {}).get("type", "")).lower() != "mask_rcnn":
        return
    if not config.get("dataset", {}).get("require_masks", True):
        return
    dataset_cfg = config["dataset"]
    dataset_root = root / dataset_cfg["root"]
    annotation_name = dataset_cfg.get("annotation_file", "_annotations.coco.json")
    failures = []
    for key in ["train", "valid", "test"]:
        split_name = dataset_cfg.get(f"{key}_split", key)
        audit = audit_mask_split(dataset_root / split_name, annotation_name, verify_images=False)
        if not audit["mask_ready"]:
            failures.append(f"{key}: missing_masks={audit['missing_masks_count']} invalid_polygons={audit['invalid_polygons_count']}")
    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(
            "Mask R-CNN requires valid COCO segmentation polygons. "
            f"Mask audit failed: {joined}. Export polygon segmentation annotations from Roboflow before training."
        )


if __name__ == "__main__":
    main()
