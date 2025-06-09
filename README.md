# 🚗 Thermal Road User Detection System

## Overview
This project was developed as part of the Automotive Innovation Challenge hosted by the Western University Engineering Faculty and sponsored by General Motors. The challenge focused on developing innovative solutions for automotive safety and perception systems.

Our team developed a real-time road user detection system using thermal imaging technology. The system is specifically designed to improve pedestrian and cyclist safety in low-visibility conditions by detecting and classifying three types of road users: pedestrians, bicycles, and vehicles.

## 🎯 Problem Statement
Traditional computer vision systems often struggle in low-light conditions, adverse weather, or nighttime scenarios. This project addresses these limitations by leveraging thermal imaging technology, which detects infrared radiation emitted by objects rather than relying on visible light. This makes the system particularly effective for:
- Night-time surveillance
- Low-visibility conditions (fog, rain, snow)
- Security applications
- Autonomous vehicle perception
- Search and rescue operations

## 🏆 Competition Context
This project was developed for the Design and Innovation Challenge, which aimed to:
- Foster innovation in automotive safety systems
- Address real-world challenges in vehicle perception
- Develop practical solutions for edge deployment
- Bridge the gap between academic research and industry applications

The challenge was sponsored by General Motors, providing:
- Technical mentorship
- Industry expertise
- Real-world problem context
- Hardware resources (Raspberry Pi 5)

## 🛠 Technical Implementation

### Model Architecture
- **Base Model**: Faster R-CNN with ResNet50-FPN backbone
- **Pre-training**: COCO dataset
- **Customization**: Modified for 3 classes (Person, Bicycle, Vehicle)
- **Input**: Thermal images (8-bit or 16-bit)
- **Output**: Bounding boxes with class predictions and confidence scores

### Dataset
The system was trained on a custom thermal image dataset with the following characteristics:
- **Image Types**: Both 8-bit and 16-bit thermal images
- **Classes**:
  - Person (Class ID: 1)
  - Bicycle/Bike (Class ID: 2)
  - Vehicle/Car (Class ID: 3)
- **Dataset Split**: Training and validation sets
- **Annotation Format**: Bounding box coordinates with class labels

### Performance Metrics
- **Mean Average Precision (mAP)**: 0.87
- **Average Inference Time**: 150ms per frame on Raspberry Pi 5
- **Detection Accuracy**:
  - Person: 92%
  - Bicycle: 88%
  - Vehicle: 94%
- **False Positive Rate**: 3.2%

## 🚀 Deployment

### Prerequisites
- Raspberry Pi 5
- Python 3.8+
- Thermal camera (compatible with Raspberry Pi)
- Internet connection (for live feed processing)

### Installation Steps

1. **Clone the Repository**
```bash
git clone [https://github.com/hAlcLeite/thermal-detection.git](https://github.com/hAlcLeite/Thermal-road-user-detection.git)
cd thermal-detection
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare Dataset**
```bash
mkdir data
# Add your thermal images to the data directory
```

4. **Run the Detection System**
```bash
./run_exec.sh --scan_path /path/to/images --output_path /path/to/output
```

### Usage
The system can be used in two modes:

1. **Live Feed Processing**
```bash
python src/detect.py --input_url <camera_url> --output_path <output_directory>
```

2. **Image Processing**
```bash
python src/detect.py --input_path <image_directory> --output_path <output_directory>
```

## 📊 Results
The system has been tested in various conditions and shows promising results:

### Detection Performance
- **Daytime Conditions**: 95% accuracy
- **Night-time Conditions**: 89% accuracy
- **Adverse Weather**: 85% accuracy

### Processing Speed
- **Raspberry Pi 5**: 6-7 FPS
- **GPU-enabled System**: 25-30 FPS

## 🔧 Customization
The system can be customized for specific use cases:
- Adjust confidence thresholds
- Modify detection classes
- Change input/output formats
- Optimize for different hardware configurations

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team
- Team 33
- Contributors: Henrique Leite, David Vera, Mohannad Salem, Arash Fallahdarrehchi

## 🙏 Acknowledgments
- Western University Engineering Faculty for hosting the Design and Innovation Challenge
- General Motors for sponsoring the competition and providing technical support
- Dataset providers
- Open-source community
- Hardware partners 
