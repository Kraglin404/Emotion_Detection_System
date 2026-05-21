import os
import numpy as np
import tensorflow as tf

H5_PATH    = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
TFLITE_OUT = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")

def convert(h5_path: str, tflite_out: str):
    print(f"[INFO] Loading model from {h5_path} …")
    model = tf.keras.models.load_model(h5_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("[INFO] Converting …")
    tflite_model = converter.convert()

    with open(tflite_out, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_out) / 1024
    print(f"[INFO] TFLite model saved → {tflite_out}  ({size_kb:.1f} KB)")

    print("[INFO] Running smoke test …")
    interpreter = tf.lite.Interpreter(model_path=tflite_out)
    interpreter.allocate_tensors()
    inp  = interpreter.get_input_details()
    outp = interpreter.get_output_details()
    dummy = np.zeros((1, 48, 48, 1), dtype=np.float32)
    interpreter.set_tensor(inp[0]["index"], dummy)
    interpreter.invoke()
    out = interpreter.get_tensor(outp[0]["index"])
    print(f"[INFO] Smoke-test output shape: {out.shape}  ✓")
    print(f"\n[INFO] Done. Copy '{tflite_out}' to ml_service/ before running the server.")

if __name__ == "__main__":
    convert(H5_PATH, TFLITE_OUT)
