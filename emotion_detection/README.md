# Emotion Detection System (PC Webcam)

Real-time facial emotion detection using your **PC webcam**.  
A CNN trained on [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) classifies emotions via a **FastAPI** REST API + **TensorFlow Lite** inference.  
Detected emotions are overlaid live on the video feed with confidence bars.

> Adapted from [riyaupadhyay1611/Edge_Cloud_Emotion_Detection_System](https://github.com/riyaupadhyay1611/Edge_Cloud_Emotion_Detection_System).  
> Raspberry Pi / PiCamera2 / OLED dependencies removed; runs on any standard PC with a webcam.

---

## Emotion Classes

| Label    | Label    | Label   |
|----------|----------|---------|
| Angry    | Disgust  | Fear    |
| Happy    | Sad      | Surprise|
| Neutral  |          |         |

---

## Architecture

```
Webcam (webcam_client.py)
        │
        │ HTTP POST /predict  { "image": "<base64 JPEG>" }
        ▼
ML Service  (FastAPI  –  ml_service/app.py)
        │
        │ TFLite inference  (emotion_model.tflite)
        ▼
Response  { "prediction": "Happy", "confidence": 0.97, "all_scores": {...} }
        │
        ▼
Live OpenCV window with bounding box + label + confidence bars
```

---

## Project Structure

```
emotion_detection/
├── ml_service/
│   ├── app.py               # FastAPI server (POST /predict)
│   └── inference.py         # TFLite inference helper (also usable standalone)
│   └── emotion_model.tflite # Converted model  ← copy here after training
├── training/
│   ├── train_model.py       # Train CNN on FER2013  → emotion_model.h5
│   ├── convert_to_tflite.py # Convert .h5 → .tflite
│   └── fer2013.csv          # Download from Kaggle  (not tracked in git)
├── webcam_client.py         # PC webcam client with live overlay
├── startml.sh               # Helper: start ML service
├── startwebcam.sh           # Helper: start webcam client
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Use a virtual environment to keep things clean:
> ```bash
> python -m venv .venv && source .venv/bin/activate   # Linux/macOS
> python -m venv .venv && .venv\Scripts\activate       # Windows
> pip install -r requirements.txt
> ```

---

### 2. Download the FER2013 dataset

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download `fer2013.csv`
3. Place it in `training/`

---

### 3. Train the model

```bash
cd training
python train_model.py        # → emotion_model.h5  (~50 epochs, EarlyStopping)
python convert_to_tflite.py  # → emotion_model.tflite
```

Then copy the TFLite file to the service folder:

```bash
cp training/emotion_model.tflite ml_service/
```

---

### 4. Start the ML service

```bash
bash startml.sh
# or manually:
uvicorn ml_service.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: http://localhost:8000/docs

---

### 5. Start the webcam client

Open a second terminal:

```bash
bash startwebcam.sh
# or manually:
python webcam_client.py --camera 0 --server http://localhost:8000/predict
```

**Controls in the video window:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot as `snapshot.jpg` |
| `p` | Pause / resume |

---

## API Reference

### `GET /health`

```json
{ "status": "ok", "emotions": ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"] }
```

### `POST /predict`

**Request body:**
```json
{ "image": "<base64-encoded JPEG>" }
```

**Response:**
```json
{
  "prediction": "Happy",
  "confidence": 0.9712,
  "all_scores": {
    "Angry": 0.0023,
    "Disgust": 0.0001,
    "Fear": 0.0045,
    "Happy": 0.9712,
    "Sad": 0.0081,
    "Surprise": 0.0134,
    "Neutral": 0.0004
  }
}
```

---

## Quick inference test (no server needed)

```bash
python ml_service/inference.py path/to/face.jpg
```

---

## Notes

- **Multiple cameras:** if `--camera 0` doesn't work, try `1` or `2`.
- **Remote server:** you can run the ML service on any machine on your network and point the client at it:  
  `python webcam_client.py --server http://192.168.1.10:8000/predict`
- **Lighter runtime:** replace `tensorflow` in `requirements.txt` with `tflite-runtime` if you only need inference (no training).
- **Training time:** FER2013 has ~35 k images; expect 10–30 min on a modern GPU, longer on CPU.
