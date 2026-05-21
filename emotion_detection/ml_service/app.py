import base64
import logging
import os
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")
IMG_SIZE   = 48
EMOTIONS   = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at '{MODEL_PATH}'.")

logger.info("Loading TFLite model ...")
interpreter = TFLiteInterpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("Model loaded")

class PredictRequest(BaseModel):
    image: str

class PredictResponse(BaseModel):
    prediction: str

def preprocess_image(b64_str: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(resized)
    
    norm    = equalized.astype(np.float32) / 255.0
    return norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def run_inference(tensor: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]

app = FastAPI(title="Emotion Detection API", version="1.0.0")

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

    logger.info(f"Prediction: {prediction}  ({confidence:.2%})")
    return PredictResponse(prediction=prediction)
