"""
app.py
Skin Disease Detection System - One-Frame Web Application
Flask backend serving the unified interactive dashboard.
"""

import os
import json
import base64
import cv2
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from preprocessor import SkinPreprocessor
from feature_extraction import SkinFeatureExtractor
from train_model import SkinModelTrainer, DISEASE_CLASSES, DISEASE_INFO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Instantiate ML & Image Processors
preprocessor = SkinPreprocessor(target_size=(256, 256))
feature_extractor = SkinFeatureExtractor()
model_trainer = SkinModelTrainer(models_dir="models")

# Global state for currently loaded image features
current_session_data = {
    "has_image": False,
    "image_filename": None,
    "features": None,
    "feature_breakdown": None,
    "active_model_algo": "random_forest"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_or_get_dataset():
    """Load or generate dataset if not existing."""
    x_path = "data/X_skin_features.npy"
    y_path = "data/y_skin_labels.npy"
    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        return X, y
    else:
        from dataset_generator import generate_benchmark_dataset
        return generate_benchmark_dataset()

@app.route('/')
def index():
    """Render the unified one-frame dashboard."""
    # Read model metadata if available
    metadata = {}
    if os.path.exists("models/model_metadata.json"):
        try:
            with open("models/model_metadata.json", "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    samples = [
        {"id": "melanoma", "name": "Melanoma (Malignant)", "file": "melanoma.jpg", "badge": "High Risk"},
        {"id": "eczema", "name": "Eczema (Dermatitis)", "file": "eczema.jpg", "badge": "Inflammatory"},
        {"id": "psoriasis", "name": "Psoriasis (Plaque)", "file": "psoriasis.jpg", "badge": "Scaly Plaque"},
        {"id": "benign_keratosis", "name": "Benign Keratosis / Mole", "file": "benign_keratosis.jpg", "badge": "Benign"},
        {"id": "acne_vulgaris", "name": "Acne Vulgaris", "file": "acne_vulgaris.jpg", "badge": "Pustular"},
        {"id": "basal_cell_carcinoma", "name": "Basal Cell Carcinoma", "file": "basal_cell_carcinoma.jpg", "badge": "Malignant"}
    ]

    return render_template(
        'index.html',
        disease_classes=DISEASE_CLASSES,
        disease_info=DISEASE_INFO,
        samples=samples,
        metadata=metadata
    )

@app.route('/api/process-image', methods=['POST'])
def process_image():
    """
    Receives an image (either uploaded file or sample selection),
    runs the full preprocessing pipeline and 42-D feature extraction.
    """
    try:
        image_bgr = None
        filename = "sample.jpg"

        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_bgr = cv2.imread(save_path)
        elif request.is_json:
            data = request.get_json()
            sample_file = data.get('sample_file')
            if sample_file:
                sample_path = os.path.join('static', 'samples', secure_filename(sample_file))
                if os.path.exists(sample_path):
                    image_bgr = cv2.imread(sample_path)
                    filename = sample_file

        if image_bgr is None:
            return jsonify({"success": False, "error": "No valid image provided or file format unsupported."}), 400

        # Step 1: Preprocessor Pipeline
        processed_data = preprocessor.process_full_pipeline(image_bgr)

        # Step 2: Feature Extraction Pipeline (42 features)
        vector, feat_dict, breakdown = feature_extractor.extract_all_features(
            processed_data["enhanced_clahe"],
            processed_data["mask_single_channel"],
            processed_data["contour"]
        )

        # Save session state
        current_session_data["has_image"] = True
        current_session_data["image_filename"] = filename
        current_session_data["features"] = vector
        current_session_data["feature_breakdown"] = breakdown

        # Convert intermediate processing stages to Base64 for one-frame UI display
        images_base64 = {
            "original": SkinPreprocessor.encode_image_base64(processed_data["original"]),
            "hair_mask": SkinPreprocessor.encode_image_base64(processed_data["hair_mask"]),
            "hair_removed": SkinPreprocessor.encode_image_base64(processed_data["hair_removed"]),
            "denoised": SkinPreprocessor.encode_image_base64(processed_data["denoised"]),
            "enhanced_clahe": SkinPreprocessor.encode_image_base64(processed_data["enhanced_clahe"]),
            "hsv_view": SkinPreprocessor.encode_image_base64(processed_data["hsv_view"]),
            "grayscale": SkinPreprocessor.encode_image_base64(processed_data["grayscale"]),
            "binary_mask": SkinPreprocessor.encode_image_base64(processed_data["binary_mask"]),
            "contour_overlay": SkinPreprocessor.encode_image_base64(processed_data["contour_overlay"])
        }

        # Calculate lesion metrics for summary banner
        cnt = processed_data["contour"]
        area_px = int(cv2.contourArea(cnt)) if cnt is not None else 0
        perimeter_px = int(cv2.arcLength(cnt, True)) if cnt is not None else 0

        return jsonify({
            "success": True,
            "filename": filename,
            "images": images_base64,
            "features_summary": {
                "total_features_count": len(vector),
                "feature_names": feature_extractor.feature_names,
                "feature_values": vector,
                "breakdown": breakdown
            },
            "lesion_metrics": {
                "area_pixels": area_px,
                "perimeter_pixels": perimeter_px,
                "asymmetry_index": feat_dict.get("Asymmetry_Index", 0.0),
                "circularity": feat_dict.get("Border_Circularity", 0.0),
                "solidity": feat_dict.get("Solidity", 0.0),
                "contrast": feat_dict.get("GLCM_Contrast", 0.0)
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Trains selected machine learning classifier (Random Forest, SVM, KNN, Logistic Regression)
    on the benchmark skin feature dataset and returns evaluation metrics & plots.
    """
    try:
        payload = request.get_json() or {}
        algorithm = payload.get("algorithm", "random_forest")
        test_size = float(payload.get("test_size", 0.25))

        # Hyperparameters
        hyperparams = {}
        if algorithm == "random_forest":
            hyperparams["n_estimators"] = int(payload.get("n_estimators", 150))
            hyperparams["max_depth"] = int(payload.get("max_depth", 12)) if payload.get("max_depth") else None
        elif algorithm == "svm":
            hyperparams["kernel"] = payload.get("kernel", "rbf")
            hyperparams["c_val"] = float(payload.get("c_val", 2.0))
        elif algorithm == "knn":
            hyperparams["n_neighbors"] = int(payload.get("n_neighbors", 5))

        X, y = load_or_get_dataset()

        # Run training and evaluation
        eval_result = model_trainer.train_and_evaluate(
            X, y,
            algorithm=algorithm,
            test_size=test_size,
            **hyperparams
        )

        current_session_data["active_model_algo"] = algorithm
        return jsonify(eval_result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """
    Predicts the skin condition for the currently extracted feature vector
    using the active trained model.
    """
    try:
        feats = current_session_data.get("features")
        if feats is None:
            return jsonify({"success": False, "error": "No image features extracted yet. Please load or upload an image first."}), 400

        algorithm = current_session_data.get("active_model_algo", "random_forest")
        model_path = os.path.join("models", f"skin_{algorithm}_model.joblib")
        scaler_path = os.path.join("models", "skin_disease_scaler.joblib")

        # Fallback to RF if specific model not found
        if not os.path.join(model_path):
            model_path = os.path.join("models", "skin_random_forest_model.joblib")
            algorithm = "random_forest"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            # Train default first
            X, y = load_or_get_dataset()
            model_trainer.train_and_evaluate(X, y, algorithm="random_forest")
            model_path = os.path.join("models", "skin_random_forest_model.joblib")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Scale feature vector
        feats_arr = np.array(feats).reshape(1, -1)
        feats_scaled = scaler.transform(feats_arr)

        # Predict
        predicted_class_idx = int(model.predict(feats_scaled)[0])
        disease_name = DISEASE_CLASSES[predicted_class_idx]

        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feats_scaled)[0]
            for idx, prob in enumerate(probs):
                c_name = DISEASE_CLASSES[idx] if idx < len(DISEASE_CLASSES) else f"Class {idx}"
                probabilities[c_name] = round(float(prob) * 100, 2)
            confidence = probabilities.get(disease_name, 90.0)
        else:
            confidence = 95.0
            probabilities[disease_name] = 95.0

        info = DISEASE_INFO.get(disease_name, {
            "severity": "Unknown",
            "urgency": "Consult physician",
            "description": "Skin lesion detected.",
            "symptoms": ["Visible lesion changes"],
            "recommendations": ["Seek clinical examination"]
        })

        return jsonify({
            "success": True,
            "disease_name": disease_name,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": algorithm.replace("_", " ").title(),
            "clinical_info": info
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/benchmark', methods=['GET'])
def benchmark():
    """Runs all 4 algorithms on the benchmark dataset for side-by-side comparison."""
    try:
        X, y = load_or_get_dataset()
        results = model_trainer.benchmark_all_models(X, y)
        return jsonify({"success": True, "benchmark": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("[SERVER] Skin Disease Detection System starting on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=True)

