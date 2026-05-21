import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_PATH   = os.path.join(os.path.dirname(__file__), "fer2013.csv")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "emotion_model.h5")
IMG_SIZE    = 48
NUM_CLASSES = 7
BATCH_SIZE  = 64
EPOCHS      = 50

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def load_fer2013(csv_path: str):
    df = pd.read_csv(csv_path)

    def parse_pixels(pixel_str):
        return np.array(pixel_str.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0

    X = np.stack(df["pixels"].apply(parse_pixels).values)
    y = tf.keras.utils.to_categorical(df["emotion"].values, NUM_CLASSES)

    if "Usage" in df.columns:
        train_mask = df["Usage"] == "Training"
        val_mask   = df["Usage"] == "PublicTest"
        test_mask  = df["Usage"] == "PrivateTest"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
    else:
        n = len(X)
        t = int(n * 0.8)
        v = int(n * 0.9)
        X_train, y_train = X[:t],    y[:t]
        X_val,   y_val   = X[t:v],   y[t:v]
        X_test,  y_test  = X[v:],    y[v:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
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

def main():
    print("[INFO] Loading FER2013 …")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(DATA_PATH)
    print(f"       Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    augment = build_augmentation()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(10_000)
        .batch(BATCH_SIZE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUT, save_best_only=True, verbose=1),
    ]

    print("\n[INFO] Training …")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    print("\n[INFO] Evaluating on test set …")
    loss, acc = model.evaluate(
        tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    )
    print(f"       Test accuracy: {acc:.4f}")
    print(f"[INFO] Model saved → {MODEL_OUT}")

if __name__ == "__main__":
    main()
