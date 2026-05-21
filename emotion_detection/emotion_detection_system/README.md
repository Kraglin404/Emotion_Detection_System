# Real-Time Facial Emotion Detection System

A low-latency, high-accuracy facial emotion detection pipeline consisting of a deep convolutional neural network, a FastAPI backend service, and an OpenCV webcam client.

---

## Architecture Overview

```text
[Webcam Client] (webcam_client.py)
       │
       │  HTTP POST /predict  {"image": "<base64-encoded JPEG>"}
       ▼
[FastAPI Service] (ml_service/app.py) ➔ TFLite (float32 Model)
       │
       │  Inference Response
       ▼
  {"prediction": "Happy"}
       │
       ▼
[Live Overlay HUD] (cv2 Bounding Box)
```

---

## Emotion Classes
The system classifies facial expressions into 7 primary categories:
* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

## Project Structure
```text
emotion_detection/
├── ml_service/
│   ├── app.py                      # FastAPI server (POST /predict)
│   └── emotion_model.tflite        # Active deployed TFLite model
├── training/
│   ├── train_from_images.py        # Custom 4-block CNN training pipeline
│   ├── evaluate.py                 # Pure-NumPy test set evaluator (7,178 images)
│   ├── reconvert.py                # Helper script to export full-precision TFLite
│   └── emotion_model.tflite        # Backup copy of active model
├── webcam_client.py                # OpenCV webcam client with live overlay HUD
├── requirements.txt                # Project dependency manifest
├── startml.sh                      # Helper shell script for ML service startup
├── startwebcam.sh                  # Helper shell script for Webcam client startup
└── README.md                       # Setup and architecture documentation
```

---

## Setup & Running the System

### 1. Install Dependencies
Initialize your virtual environment and install the required libraries:
```bash
# Initialize virtual environment
python -m venv .venv
.venv\Scripts\activate # On Windows

# Install libraries
pip install -r requirements.txt
```

### 2. Running the Server (API)
Start the FastAPI server on port `8001`:
```bash
uvicorn ml_service.app:app --host 0.0.0.0 --port 8001
```
* Interactive API Documentation can be viewed at: `http://localhost:8001/docs`
* Health check: `http://localhost:8001/health`

### 3. Running the Webcam Client
Open a second terminal window (with `.venv` active) and run:
```bash
python webcam_client.py --server http://127.0.0.1:8001/predict
```
* **Webcam Controls:**
  * Press `q` to Quit.
  * Press `s` to Save a snapshot to `snapshot.jpg`.
  * Press `p` to Pause/Resume.

### 4. Running Model Evaluation
To verify model accuracy, precision, and recall metrics over the 7,178 test images:
```bash
python training/evaluate.py
```

---

## API Documentation

### `GET /health`
* **Response:**
```json
{
  "status": "ok",
  "emotions": ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
}
```

### `POST /predict`
* **Request Body:**
```json
{
  "image": "<base64-encoded JPEG>"
}
```
* **Response Body:**
```json
{
  "prediction": "Happy"
}
```
