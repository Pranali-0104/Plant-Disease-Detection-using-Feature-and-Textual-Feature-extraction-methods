# Object-Detection-Plant-Diseases

## Run Faster R-CNN locally with this repository dataset

The original notebook uses Detectron2 and expects COCO annotation JSON files. The connected
`dataset` folder in this repository is a class-folder dataset, so use the local PyTorch script:

```powershell
python FasterRCNN\train_fasterrcnn_folder.py --epochs 5 --batch-size 2
```

For a quick setup check without training:

```powershell
python FasterRCNN\train_fasterrcnn_folder.py --dry-run --max-train-images 2 --max-valid-images 2
```

The script saves checkpoints and `classes.json` under:

```text
FasterRCNN\outputs\fasterrcnn_folder
```

Note: because the current dataset has class labels but no bounding-box annotations, the script
uses one bounding box covering the full image. For true disease-region detection, annotate the
images in COCO format and use the Detectron2 notebook.

## Run Faster R-CNN with the Roboflow COCO dataset

The `COCO DATASET` folder contains `train`, `valid`, and `test` splits with
`_annotations.coco.json` files. Use this script for a local TorchVision Faster R-CNN run:

```powershell
python FasterRCNN\train_fasterrcnn_coco.py --epochs 1 --batch-size 1 --max-train-images 2 --max-valid-images 2 --max-test-images 5
```

For a dataset health report:

```powershell
python FasterRCNN\audit_coco_dataset.py
```

For a setup-only check:

```powershell
python FasterRCNN\train_fasterrcnn_coco.py --dry-run --max-train-images 2 --max-valid-images 2 --max-test-images 5
```

Prediction previews are saved under:

```text
FasterRCNN\outputs\fasterrcnn_coco\test_predictions
```

The training script also writes:

```text
FasterRCNN\outputs\fasterrcnn_coco\metrics.csv
FasterRCNN\outputs\fasterrcnn_coco\best_checkpoint.pth
FasterRCNN\outputs\fasterrcnn_coco\last_checkpoint.pth
FasterRCNN\outputs\fasterrcnn_coco\test_metrics.json
```

Ground-truth COCO box previews can be generated with:

```powershell
python FasterRCNN\preview_coco_boxes.py --coco-json "../COCO DATASET/test/_annotations.coco.json" --image-dir "../COCO DATASET/test" --output-dir outputs/coco_test_box_previews --max-images 5
```

Compared with older Keras Faster R-CNN examples, this local version keeps the Roboflow COCO
format directly, handles train/valid/test splits, remaps COCO category ID `0` away from the
Faster R-CNN background class, saves best/last checkpoints, writes metrics, and creates clean
one-box prediction previews.

## Project Overview
This project explores the potential of deep learning in early detection and diagnosis of plant diseases—an essential step for preventing widespread crop damage and ensuring food security. Utilizing the integrated datasets from Plant Village and Plant Doc, the project features advanced object detection and instance segmentation models, including YOLOv8m, YOLOv8l, Faster-RCNN, RetinaNet, YOLOv8m-seg, YOLOv8l-seg, and Mask-RCNN. These models were assessed using precision, recall, and mean Average Precision (mAP), demonstrating deep learning's transformative capability in plant disease detection.

## Dataset Details
The final dataset consists of 3,234 images across 14 different classes of plant conditions:
1. Tomato Septoria
2. Corn Leaf Blight
3. Squash Powdery Leaf
4. Apple Healthy
5. Tomato Bacterial Spot
6. Tomato Healthy
7. Apple Rust Leaf
8. Apple Scab Leaf
9. Grape Healthy
10. Corn Rust Leaf
11. Grape Black Rot
12. Corn Gray Leaf Spot
13. BellPepper Healthy
14. BellPepper Leaf Spot

## Annotation Process
I used Roboflow to annotate the images needed for training the object detection and instance segmentation models.
- **YOLO Format:** YOLO models require annotations in YOLO format, which is a .txt file for each image specifying bounding boxes and class IDs normalized to image dimensions.
- **COCO Format:** Mask R-CNN and Faster RCNN are implemented using Detectron2. It requires annotations in COCO format, which includes JSON files detailing the images, annotations, and categories for instance segmentation and object detection.

For object detection:
- **Bounding Boxes:** Each image was annotated manually to include bounding boxes around the plant disease symptoms.
  
 ![Detection Example](https://github.com/DivyaSudagoni/Object-Detection-Plant-Diseases/blob/ab38f978d491d0ae8a3072680309584365149dac/images/obj%20detection.png))

For instance segmentation:
- **Pixel-wise Masks:** generated detailed masks for each region of interest, allowing the Mask R-CNN to perform precise segmentation at the pixel level.
  
  ![Detection Example](https://github.com/DivyaSudagoni/Object-Detection-Plant-Diseases/blob/ab38f978d491d0ae8a3072680309584365149dac/images/inst%20seg.png)




