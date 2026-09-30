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


## Usage Instructions
### Environment Setup
#### Clone the repository
git clone https://github.com/YYDS5522/yolo.git
### Dataset Statement
The agricultural pest dataset used in this project is employed solely for validating the performance of this framework and has not been used in other research. No additional citations are required.
