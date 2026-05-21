"""
app.py  –  ML Service (FastAPI)
--------------------------------
Hosts a TFLite emotion classifier on port 8000.

Endpoint:
    POST /predict
    Body:   { "image": "<base64-encoded JPEG>" }
    Response: { "prediction": "Happy", "confidence": 0.97,
                "all_scores": {"Angry": 0.01, ...} }

Start with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
or:
    bash ../startml.sh
"""

import base64
import logging
import os
from io import BytesIO
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── TFLite import (falls back gracefully) ────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")
IMG_SIZE   = 48
EMOTIONS   = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"TFLite model not found at '{MODEL_PATH}'.\n"
        "Train it first:\n"
        "  cd training && python train_model.py && python convert_to_tflite.py\n"
        "Then copy emotion_model.tflite to ml_service/."
    )

logger.info("Loading TFLite model …")
interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("Model loaded  ✓")


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    image: str   # base64-encoded JPEG/PNG


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    all_scores: Dict[str, float]


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess_image(b64_str: str) -> np.ndarray:
    """Decode base64 → OpenCV BGR → grayscale 48×48 → float32 [0,1] with batch dim."""
    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {exc}")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    norm    = resized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def run_inference(tensor: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Emotion Detection API",
    description="TFLite-powered facial emotion classifier (7 classes).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "emotions": EMOTIONS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tensor = preprocess_image(req.image)
    scores = run_inference(tensor)

    idx        = int(np.argmax(scores))
    prediction = EMOTIONS[idx]
    confidence = float(scores[idx])
    all_scores = {EMOTIONS[i]: float(scores[i]) for i in range(len(EMOTIONS))}

    logger.info(f"Prediction: {prediction}  ({confidence:.2%})")
    return PredictResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        all_scores={k: round(v, 4) for k, v in all_scores.items()},
    )
