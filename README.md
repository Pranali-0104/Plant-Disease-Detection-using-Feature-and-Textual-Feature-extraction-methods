# Deep Insights into Plant Health: A Multimodal (Visual + Textual) Approach for Precision Plant Disease Detection

[cite_start]This repository contains the B.Tech Final Year Project for our paper, **"Deep insights into Plant Health: Precision Plant Disease Detection using visual and textual feature extraction methods to overcome Agricultural barriers."** [cite: 93, 94]

[cite_start]This project was a team effort by Pranali Baviskar [cite: 99][cite_start], Aparna Warrier [cite: 100][cite_start], Shraddha Singh [cite: 101][cite_start], and Shagufta Varsi [cite: 102][cite_start], under the supervision of Dr. Jinesh Melvin[cite: 104].

[cite_start]**My specific role in this project was developing the deep learning model and performing the necessary computations**[cite: 1495].

---

### Published Paper

Our research was formally published in the **International Research Journal on Advanced Engineering and Management (IRJAEM)**.

**➡️ [View the Full Publication Here](https://goldncloudpublications.com/index.php/irjaem/article/view/873)**

---

### 1. Project Overview

This project presents a robust system for accurately detecting plant diseases by **fusing visual and textual data**.

[cite_start]Existing systems often rely *only* on visual data (a leaf image)[cite: 1124, 1125]. [cite_start]Our system improves upon this by integrating a second data stream: **textual analysis**[cite: 169]. [cite_start]It processes farmer reports, agricultural databases, and environmental data (like humidity and weather) using NLP to understand the *context* behind the visual symptoms[cite: 169, 187, 443].

[cite_start]This multimodal (Visual + Textual) approach provides a far more accurate and reliable diagnosis than a visual-only model[cite: 993, 1136].

---

### 2. System Architecture

The system operates in a multi-stage pipeline as detailed in our project report:

1.  [cite_start]**Input:** A farmer captures an image of a plant using a mobile phone or drone[cite: 406].
2.  [cite_start]**Preprocessing:** The image undergoes noise removal and resizing[cite: 408].
3.  **Dual Feature Extraction:**
    * [cite_start]**Visual (CNN):** The image is fed into CNN models (Faster R-CNN, Mask R-CNN) to extract visual features like spots, lesions, and textures[cite: 409, 412, 413, 501].
    * [cite_start]**Textual (NLP):** Environmental data (weather, soil) and textual reports are processed using **TF-IDF** and NLP models to extract key textual features[cite: 410, 505, 507].
4.  [cite_start]**Multimodal Fusion:** The visual and textual feature vectors are combined to create a single, comprehensive data point[cite: 411, 509].
5.  [cite_start]**Classification:** The fused data is fed into a classifier to identify the specific disease[cite: 414].
6.  [cite_start]**Action:** The system provides the user with real-time alerts, treatment recommendations (fertilizers, irrigation), and historical data visualization[cite: 415, 416, 417].

---

### 3. Key Project Components

This project consists of two main parts:

1.  **Backend & AI Model (This Repository):**
    * [cite_start]Contains the Python code for the **Faster R-CNN** and **Mask R-CNN** models[cite: 513, 516].
    * [cite_start]Includes the data processing scripts for the **PlantVillage dataset**[cite: 623].
    * [cite_start]Features the **multimodal fusion** logic that combines visual and NLP (TF-IDF) outputs[cite: 509, 967].
    * [cite_start]Includes a **Flask-based web application** for desktop-based recognition[cite: 848, 853].

2.  **`ApnaKhet` Mobile Application (Client):**
    * [cite_start]A user-friendly **Kotlin-based mobile app** developed by the team[cite: 595, 1497].
    * [cite_start]Allows farmers to take pictures, submit them to the backend API, and receive diagnoses and recommendations[cite: 677, 680].
    * **➡️ [View the Mobile App GitHub Repository](https://github.com/ShaguftaVarsi/ApnaKhet)**

---

### 4. Tech Stack

* [cite_start]**AI/ML:** Python, TensorFlow, PyTorch, Scikit-learn, OpenCV, Hugging Face Transformers [cite: 1448]
* [cite_start]**NLP:** NLTK, spaCy, TF-IDF [cite: 967, 1253, 1268]
* [cite_start]**Models:** CNN, Faster R-CNN, Mask R-CNN, ResNet [cite: 513, 516, 1369]
* **Backend:** Flask
* [cite_start]**Mobile App:** Kotlin [cite: 595]
* [cite_start]**Database:** MySQL [cite: 600]

---

### 5. Results & Performance

Our central hypothesis was proven correct: the multimodal (fused) model significantly outperformed visual-only baseline models.

The final evaluation, conducted on a test set of 15%, showed a **~12% improvement in F1-Score** for our multimodal system.

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| CNN (Baseline) | 82.5% | 78.9% | 80.6% |
| **Multimodal Fusion** | **93.5%** | **91.8%** | **92.6%** |

**Confusion Matrix:**
Our final model performed with high accuracy across all **38 classes** in the dataset.
<img width="783" height="763" alt="image" src="https://github.com/user-attachments/assets/99b93061-43f9-4802-ae2a-4f3564238df6" />
# Plant Disease Intelligence System

Stage 1 is a production-style Faster R-CNN disease detection baseline for Roboflow COCO datasets. Stage 2 upgrades the architecture for hybrid detection + segmentation with Mask R-CNN, lesion masks, severity estimation, and deployment-ready outputs.

## Project Structure

```text
.
├── configs/
│   └── faster_rcnn_R50_FPN.yaml
│   └── mask_rcnn_R50_FPN.yaml
├── datasets/
│   ├── coco.py
│   └── register.py
├── models/
│   ├── detector.py
│   └── trainer.py
├── inference/
│   └── predictor.py
├── utils/
│   ├── config.py
│   ├── logging.py
│   ├── metrics.py
│   ├── seed.py
│   ├── visualization.py
│   └── visualize.py
├── api/
├── notebooks/
├── outputs/
├── train.py
├── evaluate.py
├── infer.py
├── dataset_audit_masks.py
├── severity_estimator.py
├── visualize_masks.py
├── requirements.txt
└── README.md
```

## Dataset

Expected Roboflow COCO layout:

```text
COCO DATASET/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

The default config points to `COCO DATASET` and expects 38 classes.

For Mask R-CNN, the COCO annotations must include valid `segmentation` polygons. Box-only COCO exports are valid for Faster R-CNN but not sufficient for lesion segmentation or severity estimation.

## Setup

Install the base dependencies:

```powershell
pip install -r requirements.txt
```

Install Detectron2 using the official instructions for your PyTorch and CUDA version:

[Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

On a GPU machine, confirm CUDA is available:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

## Dataset Audit

Run this before training:

```powershell
python -m datasets.register --config configs/faster_rcnn_R50_FPN.yaml
```

It checks split folders, image references, invalid boxes, missing images, and class counts.

Mask-readiness audit:

```powershell
python dataset_audit_masks.py --config configs/mask_rcnn_R50_FPN.yaml
```

This checks missing masks, invalid polygons, empty annotations, and corrupted images. Mask R-CNN training intentionally fails fast if `dataset.require_masks: true` and polygons are missing.

Full image-corruption verification is available when needed:

```powershell
python dataset_audit_masks.py --config configs/mask_rcnn_R50_FPN.yaml --verify-images
```

## Visualization

Ground-truth boxes:

```powershell
python -m utils.visualize --config configs/faster_rcnn_R50_FPN.yaml --split test --ground-truth --max-images 5
```

Class distribution:

```powershell
python -m utils.visualize --config configs/faster_rcnn_R50_FPN.yaml --split train --class-distribution
```

Training loss curve after training:

```powershell
python -m utils.visualize --config configs/faster_rcnn_R50_FPN.yaml --loss-curve
```

## Training

Train Faster R-CNN R50-FPN:

```powershell
python train.py --config configs/faster_rcnn_R50_FPN.yaml
```

Train Mask R-CNN after exporting COCO polygon masks:

```powershell
python train.py --config configs/mask_rcnn_R50_FPN.yaml
```

Resume training:

```powershell
python train.py --config configs/faster_rcnn_R50_FPN.yaml --resume
```

The config controls:
- dataset paths
- number of classes
- device selection
- learning rate
- batch size
- max iterations
- checkpoint period
- validation period
- mixed precision
- early stopping

Outputs are written to:

```text
outputs/faster_rcnn_R50_FPN/
```

Key artifacts:

```text
model_final.pth
classes.json
detectron_config.yaml
metrics.csv
run.log
events.out.tfevents.*
```

## Evaluation

Evaluate on the test split:

```powershell
python evaluate.py --config configs/faster_rcnn_R50_FPN.yaml --split test
```

Evaluate Mask R-CNN:

```powershell
python evaluate.py --config configs/mask_rcnn_R50_FPN.yaml --split test
```

Evaluate a specific checkpoint:

```powershell
python evaluate.py --config configs/faster_rcnn_R50_FPN.yaml --split test --weights outputs/faster_rcnn_R50_FPN/model_final.pth
```

Metrics are saved as:

```text
outputs/<experiment>/evaluation_report.json
```

For segmentation configs with valid COCO masks, Detectron2 reports segmentation mAP. Per-mask IoU, Dice, precision, and recall helpers live in `utils/segmentation_metrics.py`.

## Inference

Single image:

```powershell
python infer.py --config configs/faster_rcnn_R50_FPN.yaml --input "COCO DATASET/test/example.jpg" --top-k 1 --score-threshold 0.5
```

Folder inference:

```powershell
python infer.py --config configs/faster_rcnn_R50_FPN.yaml --input "COCO DATASET/test" --output-dir outputs/inference --top-k 1 --score-threshold 0.5
```

Mask R-CNN inference with lesion masks and severity output:

```powershell
python infer.py --config configs/mask_rcnn_R50_FPN.yaml --input "COCO DATASET/test" --output-dir outputs/mask_inference --top-k 1 --score-threshold 0.5
```

Inference outputs:

```text
outputs/mask_inference/
├── *_prediction.jpg
├── masks/
├── overlays/
├── combined/
├── side_by_side/
└── predictions.json
└── severity_results.csv
```

Mask inference JSON is API-ready:

```json
{
  "disease_name": "Apple___Apple_scab",
  "confidence": 0.94,
  "severity_percent": 12.5,
  "severity_class": "Moderate",
  "bbox": [10.0, 20.0, 200.0, 220.0],
  "mask_path": "outputs/mask_inference/masks/example_mask.png"
}
```

## Mask Visualization

Ground-truth mask overlays:

```powershell
python visualize_masks.py --config configs/mask_rcnn_R50_FPN.yaml --split test --ground-truth --max-images 5
```

Prediction mask overlay:

```powershell
python visualize_masks.py --config configs/mask_rcnn_R50_FPN.yaml --image path/to/image.jpg --prediction-mask path/to/mask.png
```

## Severity Estimation

Severity is computed as:

```text
severity = infected_area / total_leaf_area * 100
```

The Stage 2 pipeline supports multiple lesion masks, optional foreground leaf isolation, morphological cleanup, largest-contour extraction, and configurable severity classes:

```text
Mild, Moderate, Severe, Critical
```

## Why This Baseline Is Stronger

Compared with older Keras Faster R-CNN examples, this repo uses:
- Detectron2 Faster R-CNN R50-FPN
- native COCO registration
- train/valid/test split support
- automatic class loading
- COCO pretrained weights
- GPU and mixed precision support
- checkpointing and resume
- validation loss hook
- COCO mAP evaluation
- TensorBoard logging
- clean one-box or top-k prediction visualization
- modular files instead of notebook-style scripts

## Future Extensions

The structure is ready for:
- Mask R-CNN segmentation heads
- disease severity estimation from mask or bbox area
- explainability visualizations
- weather-aware recommendations
- FastAPI model serving
- React dashboard
- mobile inference export pipeline
