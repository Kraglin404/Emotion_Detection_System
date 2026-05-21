# Real-Time Facial Emotion Detection System

A low-latency, high-accuracy facial emotion detection pipeline consisting of a deep convolutional neural network, a FastAPI backend service, and an OpenCV webcam client.

---

## System Architecture

The pipeline uses a low-latency Edge-Cloud paradigm where a lightweight client application captures real-time video frames, crops faces, and delegates heavy inference workload to a high-capacity FastAPI backend hosting an optimized TFLite model.

```text
+---------------------------------------------------------------------------------------------------+
|                                      SYSTEM PIPELINE FLOW                                         |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ WEBCAM CLIENT ]                                                                                |
|         │                                                                                         |
|         ├─► [Frame Capture] ──► [Haar Cascade Face Detection] ──► [Face Cropping]                  |
|         │                                                                                         |
|         └─► [Base64 Encoding] ───────────────────┐                                                |
|                                                  │                                                |
|  [ FASTAPI SERVER ]                              ▼  HTTP POST /predict (Base64 Image Payload)     |
|         │                                 +──────────────+                                        |
|         │                                 | FastAPI Port |                                        |
|         │                                 |     8001     |                                        |
|         │                                 +──────────────+                                        |
|         │                                        │                                                |
|         ├─► [Base64 Decoding] ──► [Grayscale CLAHE Equalization] ──► [Resizing (48x48x1)]         |
|         │                                                                                         |
|         └─► [TFLite float32 Inference] ──► [Class Mapping (ArgMax)]                                |
|                                                  │                                                |
|  [ INFERENCE HUD OVERLAY ]                       ▼ HTTP Response: {"prediction": "Emotion"}        |
|         │                                 +──────────────+                                        |
|         └─► [Draw Bounding Box] ──────────► [Draw Live Text HUD] ─────────────────────────────────┘|
|                                                                                                   |
+---------------------------------------------------------------------------------------------------+
```

---

## Convolutional Neural Network (CNN) Architecture

The deep neural network is custom-designed with 4 sequential convolutional feature extraction blocks followed by a dense classification layer, optimized specifically for abstract facial feature representations.

```text
+-----------------------------------------------------------------------------------+
|                                 CNN MODEL GRAPH                                   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  INPUT: Grayscale Face Crop [48 x 48 x 1]                                         |
|    │                                                                              |
|    ├─► [BLOCK 1] ──► Conv2D (64 filters, 3x3) ──► BatchNorm ──► Conv2D (64)        |
|    │                  ──► BatchNorm ──► MaxPooling (2x2) ──► Dropout (0.15)        |
|    │                                                                              |
|    ├─► [BLOCK 2] ──► Conv2D (128 filters, 3x3) ──► BatchNorm ──► Conv2D (128)      |
|    │                  ──► BatchNorm ──► MaxPooling (2x2) ──► Dropout (0.15)        |
|    │                                                                              |
|    ├─► [BLOCK 3] ──► Conv2D (256 filters, 3x3) ──► BatchNorm ──► Conv2D (256)      |
|    │                  ──► BatchNorm ──► MaxPooling (2x2) ──► Dropout (0.15)        |
|    │                                                                              |
|    ├─► [BLOCK 4] ──► Conv2D (512 filters, 3x3) ──► BatchNorm ──► Conv2D (512)      |
|    │                  ──► BatchNorm ──► MaxPooling (2x2) ──► Dropout (0.15)        |
|    │                                                                              |
|    ├─► [FLATTEN] ──► Flat Vector (2,048 units)                                    |
|    │                                                                              |
|    ├─► [DENSE 1] ──► Fully Connected (512 units) ──► BatchNorm ──► Dropout (0.30)  |
|    │                                                                              |
|    ├─► [DENSE 2] ──► Fully Connected (256 units) ──► BatchNorm ──► Dropout (0.30)  |
|    │                                                                              |
|    └─► [OUTPUT]  ──► Dense Softmax Classifier (7 primary classes)                 |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

### Key Architectural Enhancements:
1. **Adaptive Contrast Normalization (CLAHE):** Preprocesses faces using Contrast Limited Adaptive Histogram Equalization to mitigate severe shadowing and uneven illumination noise.
2. **Deep Abstract Filters:** A 4-block filter progression (64 → 128 → 256 → 512) to learn multi-scale spatial representations (from edges to complex expression shapes).
3. **Internal Normalization & Regularization:** Leverages `BatchNormalization` for smooth gradient flow, with conservative dropout rates (0.15 in Conv blocks, 0.30 in Dense blocks) to maximize generalization capacity without causing underfitting.

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
