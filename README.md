# Real-Time Video Surveillance for Enhancing Construction Worker Safety

![Project Banner](https://img.shields.io/badge/status-active-success.svg) ![Python](https://img.shields.io/badge/python-3.8-blue.svg) ![YOLO](https://img.shields.io/badge/YOLO-v8_to_v11-orange.svg) ![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-brightgreen.svg) ![Security](https://img.shields.io/badge/Security-Construction%20Safety-blue.svg)

This repository provides a comprehensive solution to enhance construction worker safety using real-time video surveillance, focusing on detecting critical objects such as persons, guns, and tools. This project integrates two specialized datasets to build a more robust model capable of identifying potential threats and hazardous objects in real-time.

## 🔗 Original Datasets

- **YouTube Gun Detection Dataset (YouTube-GDD)**: [Dataset Link](https://github.com/UCAS-GYX/YouTube-GDD/tree/main)
- **YOLO7 Power Tool Dataset**: [Dataset Link](https://universe.roboflow.com/keiran-gib1/yolo7-swqle)

## 📂 Dataset Overview

### RiskScanConstruction Dataset
To develop a comprehensive detection system, we combined two datasets:
1. **YouTube Gun Detection Dataset**: Contains annotated images of guns and persons in various dynamic scenarios, with rich contextual information.
2. **YOLO7 Power Tool Dataset**: Provides images of power tools commonly used in construction, grouped under a single category ("Tools") for easy detection.

#### Dataset Structure
```plaintext
RiskScanConstruction
├── train
│   ├── images
│   └── labels
├── test
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
```

Each subfolder (`train`, `test`, `valid`) includes `images` and `labels`, with label files annotated in YOLO format:
- **Class 0**: Person
- **Class 1**: Gun
- **Class 2**: Tools

## 🚀 Methodology

### 1. Dataset Preparation
We removed unlabeled images from YouTube-GDD and merged it with the YOLO7 Power Tool dataset. This final dataset comprises:
- **Training Set**: 10,953 images
- **Validation Set**: 1,213 images
- **Test Set**: 1,127 images

### 2. Model Training
We experimented with eight YOLO models (`yolov8n`, `yolov8x`, `yolov9t`, `yolov9e`, `yolov10n`, `yolov10x`, `yolo11n`, and `yolo11x`) for optimal performance. Training parameters included:
- **Epochs**: 100
- **Image Size**: 640x640
- **Batch Size**: 16
- **Early Stopping**: Patience of 30 epochs

### 3. Model Evaluation and Selection
After training, we evaluated each model on the test set with metrics:
- **Precision**: Accuracy of detection
- **Recall**: Ability to detect relevant objects
- **mAP@0.5**: Mean Average Precision at 0.5 Intersection over Union (IoU)
- **Inference Speed**: Performance efficiency for real-time use

### 4. Live Video Integration
The selected model was integrated into a live video feed for real-time detection, alerting safety personnel to potential hazards involving persons, guns, or tools on construction sites.

## 📝 Installation

To run this project, clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/Real-Time_Video_Surveillance_for_Enhancing_Construction_Worker_Safety.git
cd Real-Time_Video_Surveillance_for_Enhancing_Construction_Worker_Safety
pip install -r requirements.txt
```

## 🖥️ Usage

1. **Train the model**:
   - Place your `data.yaml` and dataset under `datasets/`.
   - Run:
     ```python
     python train.py --data data.yaml --epochs 100 --batch-size 16 --img 640
     ```

2. **Evaluate the model**:
   - Evaluate the model on the test set:
     ```python
     python val.py --data data.yaml --weights yolov8x.pt
     ```

3. **Live Video Surveillance**:
   - Integrate the trained model for real-time detection on video feeds:
     ```python
     python detect.py --source video.mp4 --weights yolov8x.pt --project Gun_Detection
     ```

## 📊 Results

- **Precision**: 0.95
- **Recall**: 0.93
- **mAP@0.5**: 0.92
- **Inference Speed**: 15 FPS

> The final model achieved high accuracy and real-time inference speeds, making it suitable for deployment in surveillance systems.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Acknowledgments

- **YouTube Gun Detection Dataset (YouTube-GDD)** for providing labeled instances of guns and persons.
- **YOLO7 Power Tool Dataset** for annotated power tool images.
  
