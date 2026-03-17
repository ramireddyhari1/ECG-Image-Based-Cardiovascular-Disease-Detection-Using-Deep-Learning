import json
import os
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from model.model_utils import CLASS_EXPLANATIONS, CLASS_NAMES, preprocess_ecg_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model", "ecg_cnn.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "saved_model", "label_map.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(
    __name__,
    template_folder=os.path.join("app", "templates"),
    static_folder=os.path.join("app", "static"),
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MODEL = None
MODEL_CLASS_NAMES = CLASS_NAMES.copy()


os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_label_map():
    if not os.path.exists(LABEL_MAP_PATH):
        return CLASS_NAMES.copy()

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    class_names = payload.get("class_names")

    if not class_names or not isinstance(class_names, list):
        return CLASS_NAMES.copy()
    return class_names


def load_model_once():
    global MODEL, MODEL_CLASS_NAMES

    if MODEL is not None:
        return MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run `python model/train_model.py` first."
        )

    MODEL = tf.keras.models.load_model(MODEL_PATH)
    MODEL_CLASS_NAMES = load_label_map()
    return MODEL


def stable_softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "ecg_image" not in request.files:
        return jsonify({"success": False, "error": "No ECG image file was uploaded."}), 400

    file = request.files["ecg_image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Please choose an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type."}), 400

    patient_name = request.form.get("patient_name", "Unknown")
    patient_age = request.form.get("patient_age", "Unknown")
    patient_gender = request.form.get("patient_gender", "Unknown")

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        model = load_model_once()
        input_tensor = preprocess_ecg_image(save_path, target_size=(224, 224))
        raw_predictions = model.predict(input_tensor, verbose=0)[0]

        if np.any(raw_predictions < 0) or not np.isclose(np.sum(raw_predictions), 1.0, atol=1e-3):
            probabilities = stable_softmax(raw_predictions)
        else:
            probabilities = raw_predictions

        predicted_idx = int(np.argmax(probabilities))
        predicted_condition = MODEL_CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        explanation = CLASS_EXPLANATIONS.get(
            predicted_condition,
            "Pattern detected. Please correlate with physician review and full ECG report.",
        )

        probability_map = {
            MODEL_CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(MODEL_CLASS_NAMES))
        }

        response = {
            "success": True,
            "image_url": f"/uploads/{unique_name}",
            "predicted_condition": predicted_condition,
            "confidence": confidence,
            "explanation": explanation,
            "probabilities": probability_map,
            "patient": {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
            },
        }
        return jsonify(response)

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
