import cv2
import numpy as np
from tensorflow.keras import layers, models

CLASS_NAMES = [
    "Normal",
    "Atrial Fibrillation",
    "Myocardial Infarction",
    "Tachycardia",
    "Bradycardia",
]

CLASS_EXPLANATIONS = {
    "Normal": "The ECG pattern appears within expected rhythm and morphology limits.",
    "Atrial Fibrillation": "Irregular rhythm and absent organized atrial activity can indicate atrial fibrillation.",
    "Myocardial Infarction": "Morphologic abnormalities may suggest ischemic injury; urgent clinical correlation is required.",
    "Tachycardia": "Heart rhythm appears faster than normal baseline and may require evaluation of underlying causes.",
    "Bradycardia": "Heart rhythm appears slower than normal baseline and should be interpreted with symptoms and history.",
}


def preprocess_ecg_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.equalizeHist(denoised)

    edges = cv2.Canny(equalized, threshold1=40, threshold2=120)
    enhanced = cv2.addWeighted(equalized, 0.85, edges, 0.15, 0)

    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    # Convert single-channel ECG to 3 channels for standard CNN input shape.
    rgb_tensor = np.stack([normalized, normalized, normalized], axis=-1)
    rgb_tensor = np.expand_dims(rgb_tensor, axis=0)
    return rgb_tensor


def build_cnn_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
