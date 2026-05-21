import sys
import os
import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")
IMG_SIZE   = 48
EMOTIONS   = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run training/train_model.py and training/convert_to_tflite.py first."
            )
        self.interpreter = TFLiteInterpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp  = self.interpreter.get_input_details()
        self.outp = self.interpreter.get_output_details()

    def predict(self, face_bgr: np.ndarray) -> dict:
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        tensor  = resized.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0

        self.interpreter.set_tensor(self.inp[0]["index"], tensor)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.outp[0]["index"])[0]

        idx   = int(np.argmax(scores))
        return {
            "emotion":    EMOTIONS[idx],
            "confidence": float(scores[idx]),
            "scores":     {EMOTIONS[i]: float(scores[i]) for i in range(len(EMOTIONS))},
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"ERROR: Cannot read image '{sys.argv[1]}'")
        sys.exit(1)

    clf    = EmotionClassifier()
    result = clf.predict(img)
    print(f"Emotion    : {result['emotion']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("All scores :")
    for k, v in sorted(result["scores"].items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 30)
        print(f"  {k:<10} {v:.4f}  {bar}")
