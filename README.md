# Enhance-YOLOv8: Agricultural Pest Detection

## Code Information
### Project Structure

ultralytics-main
├── data/
│   ├── images/                     # Folder containing dataset images
│   ├── labels/                     # Folder containing YOLO format labels
│   └── data.yaml                   # Dataset configuration file
├── ultralytics/                    # Core library directory
├── models/
│   ├── block.py                    # Implementation of Enhance_AFCA & MANet_PD modules
│   └── attention.py                # Implementation of Correlative Attention Mechanism
├── utils/
│   └── loss.py                     # Implementation of WiseIoU loss function
├── train.py                        # Model training script
├── val.py                          # Model validation script
└── get_all_yaml_param_and_flops.py # Script to calculate model parameters and GFLOPs

### Key Components

  1. Enhance_AFCA Module: Combines hierarchical multi-scale feature extraction with Adaptive Fine-grained Channel Attention mechanism
  2. MANet_PD Module: Multi-scale attention network specifically designed for pest detection
  3. WiseIoU Loss: Dynamic loss function with auxiliary bounding box and penalty mechanism

## Usage Instructions

### Requirements

  Python >= 3.10
  PyTorch >= 2.1.7
  CUDA >= 12.6
  OpenCV >= 4.8.0
  ultralytics >= 8.0.0
  numpy >= 1.24.0
  matplotlib >= 3.7.0

### Installation

# Clone the repository
  git clone https://github.com/YYDS5522/yolo.git
  cd yolo

# Install dependencies

  pip install -r requirements.txt

## Methodology

### Model Architecture

  Backbone: Modified CSPDarknet with Enhance_AFCA modules replacing C2f modules
  Neck: Enhanced feature pyramid network with MANet_PD modules
  Head: YOLOv8 detection head with WiseIoU loss function

### Training Configuration
  Hardware: NVIDIA RTX 4060 Ti (8GB), Intel i7-13620H
  Software: Windows 11, PyTorch 2.1.7, Python 3.10.18, CUDA 12.6
  Hyperparameters: SGD optimizer, lr=0.01, batch_size=8, epochs=300

