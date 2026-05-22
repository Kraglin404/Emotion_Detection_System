import os
import sys
import cv2
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    TFLiteInterpreter = tflite.Interpreter
except ImportError:
    try:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter
    except ImportError:
        print("[ERROR] Neither 'tflite_runtime' nor 'tensorflow' could be imported.")
        print("[ERROR] Please run the script using your virtual environment Python:")
        print("        .venv\\Scripts\\python training/evaluate.py")
        sys.exit(1)

IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

EMOTIONS_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def load_test_dataset(test_dir):
    X = []
    y = []
    
    if not os.path.exists(test_dir):
        return None, None
        
    categories = os.listdir(test_dir)
    for category in categories:
        cat_lower = category.lower()
        if cat_lower not in EMOTIONS_MAP:
            continue
            
        label = EMOTIONS_MAP[cat_lower]
        cat_path = os.path.join(test_dir, category)
        
        if not os.path.isdir(cat_path):
            continue
            
        img_names = os.listdir(cat_path)
        for img_name in img_names:
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
                
            X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0)
            y.append(label)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "dataset", "test")
    model_path = os.path.join(script_dir, "..", "ml_service", "emotion_model.tflite")

    if not os.path.exists(model_path):
        print(f"[ERROR] TFLite model not found at '{model_path}'.")
        print("        Please make sure you have trained and converted the model first.")
        sys.exit(1)

    interpreter = TFLiteInterpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_test, y_true = load_test_dataset(test_dir)
    if X_test is None or len(X_test) == 0:
        print(f"[ERROR] No test images found in dataset directory '{test_dir}'.")
        print("        Please ensure you have placed your test images in 'training/dataset/test/'.")
        sys.exit(1)

    y_pred = []
    
    for i in range(len(X_test)):
        tensor = X_test[i:i+1]
        interpreter.set_tensor(input_details[0]["index"], tensor)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]["index"])[0]
        y_pred.append(np.argmax(scores))
        
        if (i + 1) % 1000 == 0:
            print(f"       Evaluated {i + 1}/{len(X_test)} images ...")

    y_pred = np.array(y_pred, dtype=np.int32)

    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    accuracy = correct / total

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1

    print("\n========================================================")
    print("                EVALUATION REPORT")
    print("========================================================")
    print(f"Overall Model Accuracy: {accuracy:.2%}  ({correct}/{total} correct predictions)\n")
    
    print(f"{'Emotion':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 59)
    
    for i in range(NUM_CLASSES):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        support = np.sum(confusion[i, :])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{EMOTIONS[i]:<12} | {precision:<10.2%} | {recall:<10.2%} | {f1:<10.2%} | {support:<8}")

    print("-" * 59)

    print("\nText-based Confusion Matrix (Rows = Truth, Columns = Predicted):")
    header = "       " + "   ".join([f"{e[:3]:<3}" for e in EMOTIONS])
    print(header)
    print("       " + "-" * (len(header) - 7))
    for i in range(NUM_CLASSES):
        row_str = f"{EMOTIONS[i][:3]:<4} | " + "   ".join([f"{confusion[i, j]:<3}" for j in range(NUM_CLASSES)])
        print(row_str)
    print("========================================================")

if __name__ == "__main__":
    main()
