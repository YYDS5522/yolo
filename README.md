# Research on Agricultural Pest Detection Based on Improved YOLOv8
# Enhance-YOLOv8: A robust small-object detection framework for complex agricultural pest scenarios

## Description
This project implements the Enhance-YOLOv8 robust detection framework, specifically optimized for small‑target pest detection tasks in complex agricultural scenarios.

## Code Information
### Project Structure

```
yolo/
├── ultralytics/
│   ├── nn/
│   │   ├── modules/
│   │      └── block.py           # Core modules: Enhance_AFCA, MANet_PD, etc.
│   └── utils/
│       └── loss.py               # Loss function integration
├── data/
│   └── data.yaml                 # Dataset configuration
├── train.py                      # Training script
├── val.py                        # Validation script
└── requirements.txt              # Dependencies
```

### Core Components
The core innovative modules are implemented in models/block.py and utils/loss.py. Key components include:
### Enhance_AFCA Module:
Replaces the bottleneck structure in the standard YOLOv8 C2f module, integrating multi‑scale feature extraction with adaptive fine‑grained channel attention to enhance the capture of small‑pest features.
### MANet_PD Module: 
A multi‑scale aggregation network designed for the model neck, optimizing feature refinement and fusion via the Star_Block module.
### WiseIoU Loss Function: 
A dynamic bounding‑box regression loss with a focusing mechanism and penalty term.

## Usage Instructions

### Environment Setup

#### Clone the repository
git clone https://github.com/YYDS5522/yolo.git
cd yolo

#### Install dependencies (recommended to create a virtual environment first)
pip install -r requirements.txt

## Dataset Preparation
Place the dataset into the data/ directory following the standard YOLO structure.
Edit the data/data.yaml file to correct the train/val/test paths and add the dataset class names.

### Model Training
Execute the training script:
Basic training command (loads default configuration):
python train.py

### Model Validation/Testing
After training, evaluate the model performance using the best weights:
Basic validation command:
python val.py

## Training Configuration
### Evaluation Method

Comparative Experiments: Performance of Enhance‑YOLOv8 is compared against the YOLOv8 baseline, YOLOv5, YOLOv11, and other mainstream detectors on the same test set.
Ablation Study: Core components (Enhance_AFCA, MANet_PD, WiseIoU) are added incrementally to validate the performance contribution of each module.
Robustness Analysis: Model performance is evaluated under typical agricultural‑scene challenges such as scale variation and target occlusion.

### Assessment Metrics
Standard object‑detection metrics are used, defined as follows:
Precision: Proportion of true positives (TP) among samples predicted as positive, measuring detection accuracy.
Recall: Proportion of actual positive samples correctly detected (TP), reflecting target coverage capability.
Average Precision (AP): Integrates precision across different recall levels, measuring single‑class detection performance.
mean Average Precision (mAP): Average of AP over all classes, serving as the core metric for overall model performance.
Parameters/GFLOPs: Quantify model complexity and computational efficiency.

### Dataset Statement
The agricultural pest dataset used in this project is employed solely for validating the performance of this framework and has not been used in other research. No additional citations are required.
