# 🚁 Drone Detection System (DsDet)

A comprehensive drone detection system built with YOLOv8, featuring K-fold cross-validation training and a modern Streamlit web interface for real-time inference on both images and videos.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Web Interface](#web-interface)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a robust drone detection system using YOLOv8 with the following key components:

- **K-Fold Cross-Validation Training**: Ensures model robustness and generalization
- **Modern Web Interface**: Streamlit-based UI with file upload and real-time processing
- **Multi-format Support**: Handles both images and videos
- **Comprehensive Evaluation**: Detailed metrics and performance analysis

## ✨ Features

- 🎯 **Single-class Detection**: Specialized for drone detection
- 🔄 **K-Fold Cross-Validation**: 5-fold training for robust model evaluation
- 📊 **Real-time Metrics**: Live progress tracking and performance visualization
- 🎨 **Modern UI**: Beautiful Streamlit interface with drag-and-drop file upload
- 📹 **Video Processing**: Support for multiple video formats (MP4, AVI, MOV, MKV)
- ⚙️ **Configurable Parameters**: Adjustable confidence and IoU thresholds
- 📈 **Performance Analytics**: Comprehensive evaluation metrics

## 📁 Project Structure

```
DsDet/
├── config/
│   ├── config.yaml          # Main configuration file
│   └── dataset.yaml         # Dataset configuration
├── data/
│   ├── images/              # Training images (100+ drone images)
│   └── labels/              # YOLO format labels
├── models/
│   └── yolo/                # Model definitions
├── scripts/
│   ├── app.py              # Streamlit web interface
│   ├── train.py            # K-fold training script
│   ├── evaluate.py         # Model evaluation
│   └── predict.py          # Standalone prediction script
├── utils/
│   └── data_utils.py       # Data loading and preprocessing utilities
├── runs/
│   └── detect/             # Training results and model weights
│       ├── train_fold_0/   # Fold 0 training results
│       ├── train_fold_1/   # Fold 1 training results
│       └── ...             # Additional folds
└── output/                 # Inference results and predictions
```

## 🚀 Installation

### Prerequisites

- Python 3.9+ (recommended) or Python 3.8
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DsDet
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv DroneSwarm
   source DroneSwarm/bin/activate  # On Windows: DroneSwarm\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install ultralytics streamlit opencv-python pyyaml numpy pandas
   ```

4. **Verify installation:**
   ```bash
   python -c "import ultralytics; print('Ultralytics installed successfully')"
   ```

## 📖 Usage

### Quick Start

1. **Launch the web interface:**
   ```bash
   streamlit run scripts/app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Select a model fold** from the sidebar

4. **Upload an image or video** or provide a file path

5. **Adjust inference settings** (confidence, IoU thresholds)

6. **Run detection** and view results

### Training

```bash
python scripts/train.py
```

This will:
- Perform 5-fold cross-validation training
- Save model weights for each fold
- Generate training metrics and visualizations

### Evaluation

```bash
python scripts/evaluate.py
```

### Standalone Prediction

```bash
python scripts/predict.py
```

## 🎓 Training

### K-Fold Cross-Validation

The system uses 5-fold cross-validation to ensure robust model performance:

- **Fold 0**: Basic training (3 epochs)
- **Fold 1**: Extended training (100 epochs) - Best performance
- **Fold 2-4**: Additional validation folds

### Training Configuration

```yaml
training:
  imgsz: 640          # Input image size
  epochs: 100         # Training epochs
  batch: 16           # Batch size
  k_folds: 5          # Number of folds
  save_dir: ../output # Results directory
```

## 📊 Evaluation

### Model Performance Metrics

Based on the training results from `runs/detect/train_fold_1/` (best performing fold):

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 0.937 | Mean Average Precision at IoU=0.5 |
| **mAP50-95** | 0.638 | Mean Average Precision across IoU thresholds |
| **Precision** | 0.960 | Precision on validation set |
| **Recall** | 0.884 | Recall on validation set |
| **Training Time** | ~1007s | Total training time (100 epochs) |

### Training Progress (Fold 1)

- **Best mAP50**: 0.951 (Epoch 65)
- **Best mAP50-95**: 0.650 (Epoch 65)
- **Final Precision**: 0.960
- **Final Recall**: 0.884

### Loss Curves

The model shows excellent convergence:
- **Box Loss**: Decreased from 1.45 to 0.60
- **Classification Loss**: Decreased from 2.02 to 0.31
- **DFL Loss**: Decreased from 1.62 to 1.06

## 🔍 Inference

### Supported Formats

**Images:**
- JPG, JPEG, PNG

**Videos:**
- MP4, AVI, MOV, MKV

### Inference Parameters

```yaml
inference:
  conf: 0.25    # Confidence threshold (0.1-1.0)
  iou: 0.45     # IoU threshold (0.1-1.0)
```

### Output

- **Annotated images/videos** with bounding boxes
- **Detection statistics** (count, confidence scores)
- **JSON export** with detailed detection data

## 🌐 Web Interface

### Features

- **Modern UI**: Gradient headers and responsive design
- **File Upload**: Drag-and-drop interface
- **Progress Tracking**: Real-time processing updates
- **Metrics Display**: Live performance statistics
- **Sample Visualization**: Best detection frames for videos

### Interface Sections

1. **Model Configuration** (Sidebar)
   - Fold selection
   - Model weights (best.pt/last.pt)
   - Model status

2. **Input Data**
   - File upload area
   - Path input option
   - File validation

3. **Inference Settings**
   - Confidence threshold slider
   - IoU threshold slider

4. **Results Display**
   - Annotated images/videos
   - Detection metrics
   - Sample frames (for videos)

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
dataset:
  path: data
  images: images
  labels: labels
  names: ['drone']

model:
  name: yolov8n.pt

training:
  imgsz: 640
  epochs: 100
  batch: 16
  k_folds: 5
  save_dir: ../output

inference:
  conf: 0.25
  iou: 0.45
```

### Dataset Configuration (`config/dataset.yaml`)

```yaml
path: ../data
train: images
val: images
nc: 1
names: ['drone']
```

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'numpy._core'"**
   - **Solution**: Upgrade Python to 3.9+ and reinstall ultralytics
   ```bash
   pip install --upgrade ultralytics numpy
   ```

2. **"Can't get attribute 'DFLoss'"**
   - **Solution**: Use compatible ultralytics version with your model weights
   ```bash
   pip install "ultralytics>=8.1.0"
   ```

3. **CUDA Out of Memory**
   - **Solution**: Reduce batch size in config.yaml
   ```yaml
   training:
     batch: 8  # Reduce from 16
   ```

4. **Model Not Found**
   - **Solution**: Ensure training is complete and model weights exist
   ```bash
   python scripts/train.py
   ```

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed for faster training
- **Memory Management**: Adjust batch size based on available GPU memory
- **Data Loading**: Use SSD storage for faster data access

## 📈 Model Performance Summary

The drone detection system achieves excellent performance:

- **High Precision**: 96% precision ensures low false positives
- **Good Recall**: 88.4% recall captures most drone instances
- **Robust mAP**: 93.7% mAP50 indicates strong overall performance
- **Cross-validation**: 5-fold validation ensures generalization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: For the excellent object detection framework
- **Streamlit**: For the beautiful web interface framework
- **OpenCV**: For computer vision capabilities

---

**🚁 Happy Drone Detecting!** 🚁 