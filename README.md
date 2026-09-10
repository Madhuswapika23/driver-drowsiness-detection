---
title: DriveGuard AI - Driver Drowsiness Detection System
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# DriveGuard AI — Driver Drowsiness Detection System

**CSE303: Machine Learning | SRM University-AP**  
*Developer:* Madhu Swapnika G  

DriveGuard AI is a real-time driver drowsiness and fatigue monitoring system built with **MediaPipe FaceMesh**, **OpenCV**, and **TensorFlow/Keras**. It offers geometric landmark analysis (Eye Aspect Ratio, Mouth Aspect Ratio, PERCLOS, Head Pose) alongside a MobileNetV2 + LSTM deep learning classifier.

---

## 🌟 Key Features

- **Level 1 Heuristic Detection (`detector.py`)**: Real-time Eye Aspect Ratio (EAR) calculation using 468 3D facial landmarks from MediaPipe FaceMesh (20–30 FPS on CPU).
- **Level 2 Deep Learning & Multi-Modal Fusion (`detector_l2.py`)**: Fuses geometric indicators (EAR, MAR for yawning, PERCLOS percentage, Head Pose tilt) with a MobileNetV2 + LSTM neural network.
- **Custom Dataset & Pipeline (`collect_data.py` & `train_model.py`)**: Collect webcam eye-crop images and train lightweight CNN / LSTM / TFLite models for edge deployment.
- **Interactive Streamlit & Hugging Face Web App (`app.py`)**: Real-time WebRTC video stream processing with Web Audio API sound alerts and an Instant Camera Snapshot mode designed for cloud compatibility.
- **Multi-Platform Ready**: Optimized for desktop runtimes, Streamlit Cloud, Render, and Hugging Face Spaces.

---

## 🛠️ Project Structure

```
driver-drowsiness-detection/
├── app.py                  # Streamlit Web App (WebRTC & Cloud Snapshot modes)
├── detector.py             # Level 1 Detector (MediaPipe + EAR Heuristics)
├── detector_l2.py          # Level 2 Detector (Multi-Modal Fusion + CNN/LSTM model)
├── collect_data.py         # Eye crop dataset generator from webcam
├── train_model.py          # MobileNetV2 + LSTM classifier training & TFLite exporter
├── face_landmarker.task    # MediaPipe Face Landmarker task model bundle
├── requirements.txt        # Python dependency requirements
├── Procfile / packages.txt # Render & Cloud deployment configuration
├── dataset/                # Eye images (awake / drowsy classes)
├── model/                  # Saved models (.keras, .tflite, meta.json)
└── sounds/                 # Alarm audio generators and alert.wav
```

---

## 🚀 Quick Start (Local Desktop)

### 1. Prerequisites
- **Python 3.10** (Recommended for MediaPipe compatibility on Windows/Linux/macOS)

### 2. Installation
```bash
# Clone repository
git clone https://github.com/Madhuswapnika23/driver-drowsiness-detection.git
cd driver-drowsiness-detection

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Alert Audio
```bash
python sounds/generate_alert.py
```

### 4. Running the Detectors

#### Level 1 Detector (Fast EAR Heuristic)
```bash
python detector.py
```
*CLI Flags:*
- `--camera 1` : Select external camera source
- `--no-landmarks` : Hide facial mesh overlay
- `--no-fps` : Hide real-time FPS counter

#### Level 2 Fusion Detector (Geometric Signals + CNN/LSTM)
```bash
python detector_l2.py --mode fusion
```
*Modes available:* `fusion` (default), `cnn` (CNN only), `tflite` (TFLite edge mode).

---

## 🤖 Training Your Own Eye Classifier

1. **Collect Eye Samples**:
   ```bash
   python collect_data.py --label awake --samples 300
   python collect_data.py --label drowsy --samples 300
   ```
2. **Train & Export Models**:
   ```bash
   python train_model.py
   ```
   *Generates:* `model/cnn_eye.keras`, `model/cnn_lstm.keras`, `model/drowsiness.tflite`, and loss/accuracy plots in `model/training_plot.png`.

---

## 🌐 Web App & Cloud Deployment

### Local Streamlit Web App
```bash
streamlit run app.py
```
Access the application locally at `http://localhost:8501`.

### Deploying to Hugging Face Spaces
This project is pre-configured for **Hugging Face Spaces** (`sdk: streamlit`).

To push updates directly to Hugging Face Spaces:
```bash
git remote add hf https://huggingface.co/spaces/Madhuswapnika/driver-drowsiness-detection
git push hf main
```

---

## 📐 How It Works (Signal Pipeline)

```
Webcam Video Stream
    ├── BGR → RGB Frame Conversion
    ├── MediaPipe Face Landmarker (468 Facial Points)
    ├── Compute Key Metrics:
    │     ├── EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
    │     ├── MAR (Mouth Aspect Ratio for Yawning)
    │     ├── PERCLOS (% of time eyes closed over moving window)
    │     └── Head Pose (Pitch / Roll Angle Deviation)
    └── Level 2 Multi-Modal Fusion Engine:
          └── Combined Score = w1(EAR) + w2(MAR) + w3(PERCLOS) + w4(HeadPose) + w5(CNN/LSTM)
          └── Score > Drowsiness Threshold → Trigger Alert Synthesizer
```

---

## 📚 References & Acknowledgments
- **Soukupová & Čech (2016)** — *Real-time Eye Blink Detection using Facial Landmarks*
- **MediaPipe Vision Tasks**: [Google MediaPipe Documentation](https://google.github.io/mediapipe/)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
- **Streamlit WebRTC**: [streamlit-webrtc repository](https://github.com/whitphx/streamlit-webrtc)

