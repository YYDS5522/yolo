# Research on Agricultural Pest Detection Based on Improved YOLOv8

# Description
This project implements the Enhance-YOLOv8 robust detection framework, specifically optimized for small‑target pest detection tasks in complex agricultural scenarios. By integrating an adaptive fine‑grained channel attention module, a pest‑specific multi‑scale aggregation network, and a dynamic bounding‑box regression loss into the base YOLOv8 architecture, the framework significantly improves the detection accuracy of tiny, dense, and occluded pests in field images. The code repository includes the model implementation, training/validation scripts, and instructions for reproducing the experiments.
# Dataset Information
The performance of the framework is validated on an agricultural pest dataset. Key information about the dataset is as follows:
# Access link: 
The dataset can be obtained from: https://github.com/YYDS5522/data
# Format specification: 
Image files are in standard .jpg format, and annotation files are in YOLO‑format .txt files, both meeting human‑ and machine‑readable requirements.
# Preprocessing & Augmentation: 
The standard YOLO preprocessing pipeline is applied, specifically:
Images are uniformly resized to 640×640 pixels;
Pixel values are normalized;
Full‑epoch Mosaic augmentation is enabled during training (close_mosaic=0); other augmentation parameters follow the official YOLO default configuration (see "Training Configuration").
# Directory structure: 
The dataset follows the standard YOLO structure. The training/validation/test sets must each contain images/ and labels/ sub‑folders. Paths and class information are configured via the data/data.yaml file.

# Code Information
## Project Structure

<img width="898" height="411" alt="image" src="https://github.com/user-attachments/assets/7392574b-0e5d-4fa6-8555-4d68388b71e2" />

## Core Components
The core innovative modules are implemented in models/block.py and utils/loss.py. Key components include:
## Enhance_AFCA Module:
Replaces the bottleneck structure in the standard YOLOv8 C2f module, integrating multi‑scale feature extraction with adaptive fine‑grained channel attention to enhance the capture of small‑pest features.
## MANet_PD Module: 
A multi‑scale aggregation network designed for the model neck, optimizing feature refinement and fusion via the Star_Block module.
## WiseIoU Loss Function: 
A dynamic bounding‑box regression loss with a focusing mechanism and penalty term, tailored for regression tasks on low‑quality samples in pest datasets.

# Usage Instructions

## Environment Setup

### Clone the repository
git clone https://github.com/YYDS5522/yolo.git
cd yolo

### Install dependencies (recommended to create a virtual environment first)
pip install -r requirements.txt

## Installation

### Clone the repository
  git clone https://github.com/YYDS5522/yolo.git
  cd yolo

### Install dependencies

  pip install -r requirements.txt

## Dataset Preparation
Place the dataset into the data/ directory following the standard YOLO structure.
Edit the data/data.yaml file to correct the train/val/test paths and add the dataset class names.

# Model Training
Execute the training script:
Basic training command (loads default configuration):
python train.py

# Model Validation/Testing
After training, evaluate the model performance using the best weights:
Basic validation command:
python val.py

# Requirements
Running this project requires the following environment dependencies:
Python ≥ 3.10.18
PyTorch ≥ 2.1.7 (GPU version must match the corresponding CUDA; cu126 is recommended)
ultralytics ≥ 8.2.50
opencv‑python ≥ 4.8.0
numpy ≥ 1.24.0
matplotlib ≥ 3.7.0

# Methodology
Model Architecture
Backbone: Improved CSPDarknet, where the standard C2f modules are replaced with Enhance_AFCA modules to enhance multi‑scale contextual feature extraction for small targets.
Neck: MANet_PD module, designed based on the MANet module, optimizes multi‑scale feature fusion and pest feature representation.
Head: The standard YOLOv8 detection head is retained, but trained with the WiseIoU loss function to improve bounding‑box regression accuracy.

## Training Configuration
# Key Hyperparameters
Optimizer: SGD
Batch Size: 8; Epochs: 300; Early‑stopping Patience: 100
Input Image Size: 640×640
Loss Function Coefficients: iou=0.7

# Evaluation Method

Comparative Experiments: Performance of Enhance‑YOLOv8 is compared against the YOLOv8 baseline, YOLOv5, YOLOv11, and other mainstream detectors on the same test set.
Ablation Study: Core components (Enhance_AFCA, MANet_PD, WiseIoU) are added incrementally to validate the performance contribution of each module.
Robustness Analysis: Model performance is evaluated under typical agricultural‑scene challenges such as scale variation and target occlusion.

# Assessment Metrics
Standard object‑detection metrics are used, defined as follows:
Precision: Proportion of true positives (TP) among samples predicted as positive, measuring detection accuracy.
Recall: Proportion of actual positive samples correctly detected (TP), reflecting target coverage capability.
Average Precision (AP): Integrates precision across different recall levels, measuring single‑class detection performance.
mean Average Precision (mAP): Average of AP over all classes, serving as the core metric for overall model performance.
Parameters/GFLOPs: Quantify model complexity and computational efficiency.

# Dataset Statement
The agricultural pest dataset used in this project is employed solely for validating the performance of this framework and has not been used in other research. No additional citations are required.






