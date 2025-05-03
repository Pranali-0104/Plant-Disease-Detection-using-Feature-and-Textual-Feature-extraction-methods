# Object-Detection-Plant-Diseases

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




