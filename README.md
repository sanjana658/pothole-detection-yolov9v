# Multi-Weather Pothole Detection Using YOLOv9 with XAI and VLM

## Overview

This project focuses on detecting potholes under different weather and lighting conditions using YOLOv9.

The project is based on the Multi-Weather Pothole Detection (MWPD) dataset and follows a two-model experimental setup:

- **Model 1:** Standard YOLOv9-c baseline
- **Model 2:** Enhanced YOLOv9-c with selected ADown layers replaced by standard convolution layers

After detection, **Grad-CAM-based Explainable AI (XAI)** is used to visualize where the model focuses. **Qwen3-VL-4B-Instruct** is then used for visual analysis of detected potholes, including visible severity and safety-risk assessment.

## Dataset

**Multi-Weather Pothole Detection (MWPD)**

Current dataset split used in this project:

- Training images: 2,730
- Validation images: 260
- Test images: 97
- Classes: 1 (`Potholes`)

The images cover different weather and lighting conditions.

## Models

### Model 1 — YOLOv9-c Baseline

Standard YOLOv9-c was trained on the MWPD dataset.

Training configuration:

- Epochs: 100
- Batch size: 4
- Image size: 640 × 640
- Optimizer: SGD
- GPU: NVIDIA Tesla T4

Validation results:

| Metric | Result |
|---|---:|
| Precision | 83.1% |
| Recall | 77.6% |
| mAP@50 | 81.7% |
| mAP@50-95 | 45.3% |

### Model 2 — Enhanced YOLOv9-c

The enhanced architecture follows the selected modification from the base paper, replacing five selected ADown layers with standard convolution layers.

The same dataset and major training settings were used for comparison.

Validation results:

| Metric | Result |
|---|---:|
| Precision | 78.9% |
| Recall | 78.1% |
| mAP@50 | 80.5% |
| mAP@50-95 | 44.0% |

Under the current experimental setup, the baseline YOLOv9-c performed slightly better overall on the validation set.

## XAI — Grad-CAM

Grad-CAM was applied to the enhanced YOLOv9 model to visualize the regions contributing to a pothole detection.

The XAI stage helps interpret **where the model is focusing** when making a detection.

## VLM Analysis

The project also uses **Qwen3-VL-4B-Instruct** for visual analysis of detected potholes.

The VLM is used to provide:

- Visual description
- Pothole characteristics
- Approximate visible severity
- Potential safety risk
- Reasoning based on visible features such as size, water, cracks, and debris

An external pothole image was also tested using the complete **YOLOv9 + XAI + VLM** pipeline.

## Project Pipeline

```text
Input Road Image
       ↓
Preprocessing
       ↓
MWPD Dataset
       ↓
YOLOv9 Pothole Detection
       ↓
Best Detection Model
       ↓
Grad-CAM XAI
       ↓
Qwen3-VL Analysis
       ↓
Severity / Risk Assessment
