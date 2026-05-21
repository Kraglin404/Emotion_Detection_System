import os
import tensorflow as tf

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saved_model_path = os.path.join(script_dir, "saved_model")
    tflite_out = os.path.join(script_dir, "emotion_model.tflite")
    ml_service_out = os.path.join(script_dir, "..", "ml_service", "emotion_model.tflite")

    print(f"\n[INFO] Re-converting SavedModel at {saved_model_path} to TFLite without optimizations...")
    if not os.path.exists(saved_model_path):
        print(f"[ERROR] SavedModel path does not exist: {saved_model_path}")
        return

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    print("[INFO] Converting to TFLite ...")
    tflite_model = converter.convert()

    with open(tflite_out, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] TFLite saved to training folder -> {tflite_out}")

    os.makedirs(os.path.dirname(ml_service_out), exist_ok=True)
    with open(ml_service_out, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Copy successfully saved to ml_service folder -> {ml_service_out}")
    print("[SUCCESS] Full-precision model conversion complete!")

if __name__ == "__main__":
    main()
