import os
import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 48
NUM_CLASSES = 7
BATCH_SIZE = 64

EMOTIONS_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6
}

def load_dataset_from_directory(base_dir, max_samples=None):
    X = []
    y = []
    
    categories = os.listdir(base_dir)
    for category in categories:
        cat_lower = category.lower()
        if cat_lower not in EMOTIONS_MAP:
            continue
            
        label = EMOTIONS_MAP[cat_lower]
        cat_path = os.path.join(base_dir, category)
        
        if not os.path.isdir(cat_path):
            continue
            
        img_names = os.listdir(cat_path)
        if max_samples:
            img_names = img_names[:max_samples]
            
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
            
    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(y), NUM_CLASSES)
    return X, y

def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.15),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.15),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.15),

        layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.15),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.30),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model

def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ])

def convert_and_copy_tflite(saved_model_path, tflite_out, ml_service_out):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()

    with open(tflite_out, "wb") as f:
        f.write(tflite_model)

    os.makedirs(os.path.dirname(ml_service_out), exist_ok=True)
    with open(ml_service_out, "wb") as f:
        f.write(tflite_model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    is_fast_mode = not args.full

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        sys.exit(1)

    max_samples = 200 if is_fast_mode else None
    epochs = 2 if is_fast_mode else 40

    X_train, y_train = load_dataset_from_directory(train_dir, max_samples)
    X_test, y_test = load_dataset_from_directory(test_dir, max_samples)

    if len(X_train) == 0:
        sys.exit(1)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    saved_model_path = os.path.join(script_dir, "saved_model")
    tflite_out = os.path.join(script_dir, "emotion_model.tflite")
    ml_service_out = os.path.join(script_dir, "..", "ml_service", "emotion_model.tflite")

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    ]

    if is_fast_mode:
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=BATCH_SIZE)
        model.export(saved_model_path, format="tf_saved_model")
    else:
        augment = build_augmentation()
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10_000)
            .batch(BATCH_SIZE)
            .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        model.fit(train_ds, validation_data=(X_test, y_test), epochs=epochs, callbacks=callbacks)
        model.export(saved_model_path, format="tf_saved_model")

    convert_and_copy_tflite(saved_model_path, tflite_out, ml_service_out)

if __name__ == "__main__":
    main()
