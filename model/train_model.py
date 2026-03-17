import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from model.model_utils import build_cnn_model


EXPECTED_CLASSES = [
    "Normal",
    "Atrial Fibrillation",
    "Myocardial Infarction",
    "Tachycardia",
    "Bradycardia",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train ECG CNN model from image folders")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/ecg_images",
        help="Path to dataset root with subfolders for each class",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="model/saved_model/ecg_cnn.keras",
        help="Path to save trained model",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Square image size")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def plot_history(history, output_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=args.val_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    val_ds = keras.utils.image_dataset_from_directory(
        args.dataset_dir,
        validation_split=args.val_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    missing = [c for c in EXPECTED_CLASSES if c not in class_names]
    if missing:
        raise ValueError(
            "Dataset class folders are missing required classes: " + ", ".join(missing)
        )

    print("Detected classes:", class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
        ]
    )

    normalize = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalize(data_augmentation(x)), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    model = build_cnn_model(input_shape=(args.img_size, args.img_size, 3), num_classes=len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    training_callbacks = [
        callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=training_callbacks,
    )

    model.save(args.output_model)

    label_map_path = os.path.join(os.path.dirname(args.output_model), "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": class_names}, f, indent=2)

    metrics_plot_path = os.path.join(os.path.dirname(args.output_model), "training_metrics.png")
    plot_history(history, metrics_plot_path)

    print(f"Training complete. Model saved to: {args.output_model}")
    print(f"Label map saved to: {label_map_path}")
    print(f"Training curves saved to: {metrics_plot_path}")


if __name__ == "__main__":
    main()
