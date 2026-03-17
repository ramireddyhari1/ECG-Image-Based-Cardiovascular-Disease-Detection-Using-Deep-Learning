# ECG Image-Based Cardiovascular Disease Detection Using Deep Learning

A complete Flask-based full-stack application for classifying ECG images into:

- Normal
- Atrial Fibrillation
- Myocardial Infarction
- Tachycardia
- Bradycardia

The project includes:

- CNN training pipeline (TensorFlow/Keras)
- OpenCV preprocessing
- Flask inference API
- Modern HTML/CSS/JavaScript frontend
- ECG waveform visualization
- Confidence probability chart
- Patient report generation

## 1. Tech Stack

- Python
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy
- Chart.js (frontend visualization)

## 2. Folder Structure

```text
ecg-cvd-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── main.js
│   └── uploads/
│       └── .gitkeep
├── model/
│   ├── __init__.py
│   ├── model_utils.py
│   ├── train_model.py
│   └── saved_model/
│       └── .gitkeep
└── data/
    └── README.md
```

## 3. Dataset Setup

Use ECG image datasets built from public sources such as PhysioNet MIT-BIH Arrhythmia data. For the five target classes, keep your image dataset in this format:

```text
data/ecg_images/
├── Normal/
├── Atrial Fibrillation/
├── Myocardial Infarction/
├── Tachycardia/
└── Bradycardia/
```

Each folder should contain ECG image files (`.png`, `.jpg`, etc).

Notes:

- MIT-BIH is commonly used for arrhythmia classes.
- For robust Myocardial Infarction class coverage, many projects combine additional PhysioNet/PTB-style ECG resources.

## 4. Installation

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Train the CNN Model

```bash
python model/train_model.py --dataset_dir data/ecg_images --epochs 30 --batch_size 32
```

This creates:

- `model/saved_model/ecg_cnn.keras`
- `model/saved_model/label_map.json`
- `model/saved_model/training_metrics.png`

## 6. Run the Flask Application

```bash
python app.py
```

Open in browser:

- `http://127.0.0.1:5000`

## 7. How Prediction Works

1. Upload ECG image from the web UI.
2. Backend preprocesses image using OpenCV:
   - Grayscale conversion
   - Gaussian denoising
   - Histogram equalization
   - Edge-enhanced contrast
   - Resize + normalize
3. CNN model predicts class probabilities.
4. UI displays:
   - Uploaded ECG image
   - Predicted disease
   - Confidence score
   - Basic medical explanation
   - Probability chart
   - Waveform visualization
   - Downloadable patient report

## 8. API Endpoint

### `POST /predict`

Form-data:

- `ecg_image` (required)
- `patient_name` (optional)
- `patient_age` (optional)
- `patient_gender` (optional)

Response (JSON):

```json
{
  "success": true,
  "image_url": "/uploads/<file>",
  "predicted_condition": "Atrial Fibrillation",
  "confidence": 0.93,
  "explanation": "...",
  "probabilities": {
    "Normal": 0.02,
    "Atrial Fibrillation": 0.93,
    "Myocardial Infarction": 0.01,
    "Tachycardia": 0.03,
    "Bradycardia": 0.01
  },
  "patient": {
    "name": "...",
    "age": "...",
    "gender": "..."
  }
}
```

## 9. Medical Disclaimer

This software is for educational and research support only and is not a substitute for diagnosis by a licensed cardiologist.
