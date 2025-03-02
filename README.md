# Automatic Number Plate Recognition (ANPR) System

You can test the system directly via our Hugging Face Space:
👉 [ANPR System - Hugging Face Demo](https://huggingface.co/spaces/mchundi/Group-15-ANPR-Yolo11s-PaddleOCR)
This demo allows you to upload an image of a vehicle with a visible license plate and view the detected plate along with the recognized characters in real-time.

## Overview
This project implements an **Automatic Number Plate Recognition (ANPR) System** leveraging **YOLOv11** for license plate detection and **PaddleOCR** for character recognition. The system is designed to efficiently recognize Indian vehicle license plates under diverse real-world conditions, making it suitable for traffic management, law enforcement, and parking enforcement applications.

## Features
- **State-of-the-art deep learning models** for plate detection and character recognition.
- **Optimized image preprocessing** for handling challenging conditions (low-light, skewed angles, occlusions).
- **High accuracy performance** with **92.3% mAP@0.5** for detection and **89.1% OCR accuracy**.
- **Efficient processing speed** (~150ms per image), optimized for real-time applications.
- **Modular and scalable architecture**, allowing further improvements and real-world deployment.
- **Deployed on Hugging Face and local servers** for easy accessibility.

## Methodology
### 1. Data Collection
- Dataset sources: **Google Images and OLX (state-wise dataset)**
- Total dataset: **800 training images, 50 validation images, and a test set**
- Images are labeled with bounding boxes using XML annotation files.

### 2. Tools and Technologies
- **Deep Learning Framework:** PyTorch (with CUDA acceleration)
- **Detection Model:** YOLOv11 (YOLOv11s variant for efficiency)
- **OCR Engine:** PaddleOCR
- **Preprocessing Libraries:** OpenCV, NumPy
- **Visualization:** Matplotlib
- **Development Environment:** Kaggle (NVIDIA Tesla P100 GPU)

### 3. Implementation Steps
#### a) Data Preprocessing
- XML parsing for extracting bounding boxes.
- Image normalization and augmentation (rotation, scaling, shearing).
- Data validation checks to ensure correct format and consistency.

#### b) Model Training
- **YOLOv11 Training:**
  - 200 epochs, batch size of 16.
  - Optimized with **AdamW optimizer** and **cosine decay learning rate**.
  - Implemented **data augmentation** for better generalization.

- **OCR (PaddleOCR) Training:**
  - Grayscale conversion and adaptive thresholding for binarization.
  - Noise reduction to improve character recognition accuracy.

#### c) License Plate Detection & OCR
- YOLOv11 detects license plate regions in images.
- Detected plates are extracted and passed to PaddleOCR.
- Preprocessing applied: grayscale conversion, thresholding, and denoising.
- OCR extracts and verifies recognized characters.

#### d) Visualization & Logging
- Results visualized with bounding boxes and recognized text overlay.
- Performance monitored via logs capturing **detection time, OCR accuracy, and errors**.

### 4. Experimental Setup
- **Hardware:** NVIDIA Tesla P100 GPU (CUDA 11.2), 16GB RAM.
- **Performance Metrics:**
  - **Detection:** Mean Average Precision (mAP@0.5) - **92.3%**
  - **OCR Accuracy:** Character-wise - **89.1%**, Word-wise - **85.4%**
  - **Processing Time:** ~150ms per image.
- Evaluated across different lighting conditions and plate designs.

## Results
- **High detection and recognition accuracy** (exceeding benchmarks of comparable systems).
- **Adaptive thresholding improved low-light accuracy by 18.5%.**
- **Optimized batch processing reduced latency by 25%.**
- **Model performed well under diverse conditions, including occlusions and angle distortions.**

## Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| Poor low-light performance | Adaptive thresholding & noise reduction |
| Blurry and occluded plates | Improved preprocessing & denoising techniques |
| Confusion between similar characters (O/0, 8/B) | Context-aware OCR filtering |
| High processing latency | Optimized GPU utilization & batch processing |
| Variability in plate formats | Augmented dataset with diverse samples |

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision torchaudio
pip install paddleocr
pip install opencv-python numpy matplotlib
```

### Running the Model
1. Clone the repository:
```bash
git clone https://github.com/muralich2002/ANPR-System.git
cd ANPR-System
```
2. Run the ANPR pipeline:
```bash
python anpr_pipeline.py --input sample_image.jpg
```
3. View results in the **output/** directory.

## Hyperparameter Tuning - Recommendations
| Parameter | Original | Optimized | Benefits |
|-----------|---------|-----------|----------|
| Batch Size | 32 | 16 | Reduced overfitting, better generalization |
| Image Resolution | 640px | 768px | Improved feature extraction for small text |
| Optimizer | Adam | AdamW | Better weight decay regularization |
| Learning Rate | 0.01 | 0.005 | Prevents large weight fluctuations |
| Rotation Range | ±15° | ±30° | Better handling of skewed plates |
| Shear Transform | 2.0 | 5.0 | Improved robustness to distortions |

## Future Enhancements
- **Real-time video processing** for surveillance applications.
- **Mobile application integration** for on-the-go ANPR.
- **Adaptive learning techniques** to improve performance in low-light and poor-quality images.
- **Support for multi-lingual license plates.**

## References
1. R. Laroca et al., “A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector.”
2. M. A. Rafique et al., “Vehicle license plate detection using region-based convolutional neural networks.”
3. Tejas Thapliyala et al., “Automatic License Plate Recognition using YOLOv5 and Tesseract OCR.”
4. R. Madhumitha, “Efficient Vehicle Registration Recognition System.”


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

