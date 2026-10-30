"""
train_model.py
Skin Disease Detection - Machine Learning Training & Evaluation Engine

Supports training, cross-validation, and performance evaluation for:
- Random Forest Classifier
- Support Vector Machine (SVM - RBF/Linear)
- K-Nearest Neighbors (KNN)
- Logistic Regression

Generates:
- Confusion Matrix heatmap
- Multi-class metrics (Accuracy, Precision, Recall, F1-Score)
- Per-class classification reports
- Feature Importance rankings
- Cross-model benchmark comparison
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for web server
import matplotlib.pyplot as plt
import io
import base64

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Disease Classes Metadata
DISEASE_CLASSES = [
    "Melanoma",
    "Eczema",
    "Psoriasis",
    "Benign Keratosis",
    "Acne Vulgaris",
    "Basal Cell Carcinoma"
]

DISEASE_INFO = {
    "Melanoma": {
        "severity": "High (Malignant)",
        "urgency": "Immediate Oncological Consultation Required",
        "description": "A serious form of skin cancer that begins in cells known as melanocytes. Highly treatable when detected early.",
        "symptoms": ["Asymmetrical irregular borders", "Color variation (black, brown, red)", "Diameter > 6mm", "Evolving shape or size"],
        "recommendations": ["Urgent biopsy by a board-certified dermatologist", "Avoid sun exposure", "Do not attempt home remedies"]
    },
    "Eczema": {
        "severity": "Low to Moderate (Chronic Inflammatory)",
        "urgency": "Routine Dermatological Care",
        "description": "An inflammatory condition that causes itchy, dry, cracked, and red or inflamed skin.",
        "symptoms": ["Intense pruritus (itching)", "Erythematous dry patches", "Lichenification from scratching", "Sensitive skin flare-ups"],
        "recommendations": ["Use thick fragrance-free emollients", "Mild topical hydrocortisone as prescribed", "Avoid hot showers and harsh soaps"]
    },
    "Psoriasis": {
        "severity": "Moderate (Autoimmune Chronic)",
        "urgency": "Dermatological Consultation Recommended",
        "description": "An autoimmune skin disease where skin cells multiply up to 10 times faster than normal, building up into bumpy red patches covered with white scales.",
        "symptoms": ["Plaques covered with silvery scales", "Dry, cracked skin that may bleed", "Burning or soreness", "Pitted nails"],
        "recommendations": ["Topical corticosteroids or vitamin D analogues", "Phototherapy under medical supervision", "Systemic biologic treatments for severe cases"]
    },
    "Benign Keratosis": {
        "severity": "Low (Non-cancerous)",
        "urgency": "Elective / Monitoring",
        "description": "A common, harmless, non-cancerous skin growth that often appears as a waxy brown, black, or tan growth (seborrheic keratosis or benign melanocytic nevus).",
        "symptoms": ["Round or oval-shaped spot", "Waxy, stuck-on appearance", "Uniform coloration", "Painless, well-defined border"],
        "recommendations": ["Self-monitor for changes in size, shape, or color", "Routine annual skin check", "Can be removed for cosmetic reasons if desired"]
    },
    "Acne Vulgaris": {
        "severity": "Low to Moderate",
        "urgency": "Routine Care",
        "description": "A common skin condition that happens when hair follicles become plugged with oil, dead skin cells, and Cutibacterium acnes bacteria.",
        "symptoms": ["Papules, pustules, comedones (whiteheads/blackheads)", "Inflamed red bumps", "Sebum excess in T-zone", "Occasional tender nodules"],
        "recommendations": ["Topical benzoyl peroxide, salicylic acid, or retinoids", "Gentle non-comedogenic cleanser twice daily", "Consult doctor for persistent cystic acne"]
    },
    "Basal Cell Carcinoma": {
        "severity": "Moderate to High (Locally Invasive)",
        "urgency": "Prompt Medical Attention Required",
        "description": "The most common form of skin cancer arising in the basal cells. Rarely metastasizes but can cause local tissue destruction if untreated.",
        "symptoms": ["Pearly or waxy bump", "Visible tiny blood vessels (telangiectasia)", "Bleeding or scabbing lesion that repeatedly heals and reopens", "Rolled pearly border"],
        "recommendations": ["Surgical excision, Mohs surgery, or cryotherapy", "Full body dermatological screening", "Strict daily UV protection (SPF 50+)"]
    }
}

class SkinModelTrainer:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.scaler_path = os.path.join(models_dir, "skin_disease_scaler.joblib")
        self.metadata_path = os.path.join(models_dir, "model_metadata.json")

    def build_classifier(self, algorithm="random_forest", **kwargs):
        """Construct classifier based on requested algorithm."""
        if algorithm == "random_forest":
            n_estimators = int(kwargs.get("n_estimators", 150))
            max_depth = int(kwargs.get("max_depth", 12)) if kwargs.get("max_depth") else None
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == "svm":
            kernel = kwargs.get("kernel", "rbf")
            c_val = float(kwargs.get("c_val", 1.0))
            return SVC(
                C=c_val,
                kernel=kernel,
                probability=True,
                random_state=42
            )
        elif algorithm == "knn":
            n_neighbors = int(kwargs.get("n_neighbors", 5))
            weights = kwargs.get("weights", "distance")
            return KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights
            )
        elif algorithm == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train_and_evaluate(self, X, y, algorithm="random_forest", test_size=0.25, **hyperparams):
        """
        Executes full training, test evaluation, and metric calculation.
        """
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and train model
        model = self.build_classifier(algorithm, **hyperparams)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

        # Primary Performance Metrics
        acc = float(accuracy_score(y_test, y_pred))
        precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # 5-Fold Cross Validation
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

        # Classification report dict
        unique_labels = sorted(list(set(y)))
        target_names = [DISEASE_CLASSES[i] if i < len(DISEASE_CLASSES) else f"Class {i}" for i in unique_labels]
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)

        # Feature importances if applicable
        feature_importances = []
        if hasattr(model, "feature_importances_"):
            from feature_extraction import SkinFeatureExtractor
            feat_names = SkinFeatureExtractor().feature_names
            importances = model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            for idx in sorted_indices[:15]:
                feature_importances.append({
                    "name": feat_names[idx] if idx < len(feat_names) else f"Feature {idx}",
                    "importance": round(float(importances[idx]), 4)
                })

        # Save model and scaler to disk
        model_filename = f"skin_{algorithm}_model.joblib"
        model_full_path = os.path.join(self.models_dir, model_filename)
        joblib.dump(model, model_full_path)
        joblib.dump(scaler, self.scaler_path)

        # Save active model metadata
        metadata = {
            "algorithm": algorithm,
            "accuracy": round(acc * 100, 2),
            "precision_weighted": round(precision_weighted * 100, 2),
            "recall_weighted": round(recall_weighted * 100, 2),
            "f1_weighted": round(f1_weighted * 100, 2),
            "cv_mean": round(cv_mean * 100, 2),
            "cv_std": round(cv_std * 100, 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": X.shape[1],
            "model_file": model_filename
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Generate base64 Confusion Matrix and ROC plots
        cm_base64 = self.generate_confusion_matrix_plot(cm, target_names)
        roc_base64 = self.generate_roc_plot(y_test, y_prob, target_names) if y_prob is not None else ""

        return {
            "success": True,
            "algorithm": algorithm,
            "metrics": {
                "accuracy": round(acc * 100, 2),
                "precision": round(precision_weighted * 100, 2),
                "recall": round(recall_weighted * 100, 2),
                "f1_score": round(f1_weighted * 100, 2),
                "precision_macro": round(precision_macro * 100, 2),
                "recall_macro": round(recall_macro * 100, 2),
                "f1_macro": round(f1_macro * 100, 2),
                "cv_mean": round(cv_mean * 100, 2),
                "cv_std": round(cv_std * 100, 2),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": [[round(float(v), 2) for v in row] for row in cm_norm],
            "class_names": target_names,
            "classification_report": report,
            "feature_importances": feature_importances,
            "confusion_matrix_img": cm_base64,
            "roc_curve_img": roc_base64
        }

    def generate_confusion_matrix_plot(self, cm, class_names):
        """Generates sleek, dark-themed Confusion Matrix visualization base64 image."""
        plt.figure(figsize=(7, 6), facecolor="#1e293b")
        ax = plt.subplot(111)
        ax.set_facecolor("#0f172a")

        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.9)
        plt.colorbar(cax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha='left', color="#cbd5e1", fontsize=9)
        ax.set_yticklabels(class_names, color="#cbd5e1", fontsize=9)

        # Annotate text
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(j, i, f"{val}",
                        ha="center", va="center",
                        color="white" if val > thresh else "#0f172a",
                        fontweight="bold", fontsize=11)

        ax.set_xlabel('Predicted Diagnostic Label', color="#f8fafc", fontsize=11, labelpad=10)
        ax.set_ylabel('True Dermatological Label', color="#f8fafc", fontsize=11)
        ax.set_title('Performance Confusion Matrix', color="#38bdf8", fontsize=13, pad=20, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
        plt.close()
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    def generate_roc_plot(self, y_test, y_prob, class_names):
        """Generates Multi-class ROC Curve plot base64 image."""
        n_classes = len(class_names)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        if y_test_bin.shape[1] == 1 and n_classes == 2:
            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

        plt.figure(figsize=(7, 5), facecolor="#1e293b")
        ax = plt.subplot(111)
        ax.set_facecolor("#0f172a")

        colors = ['#38bdf8', '#4ade80', '#f43f5e', '#fbbf24', '#a855f7', '#ec4899']
        for i in range(min(n_classes, y_prob.shape[1])):
            if i < y_test_bin.shape[1]:
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                        label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], color='#64748b', lw=1.5, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', color="#f8fafc", fontsize=10)
        ax.set_ylabel('True Positive Rate (Sensitivity)', color="#f8fafc", fontsize=10)
        ax.set_title('Multi-Class ROC Performance Curves', color="#38bdf8", fontsize=12, pad=12, fontweight="bold")
        ax.tick_params(colors="#cbd5e1")
        ax.grid(color="#334155", linestyle=":", alpha=0.6)
        legend = ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569")
        for text in legend.get_texts():
            text.set_color("#f8fafc")
            text.set_fontsize(8)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, facecolor=plt.gcf().get_facecolor(), edgecolor='none')
        plt.close()
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

    def benchmark_all_models(self, X, y):
        """
        Runs comprehensive benchmark comparing Random Forest, SVM, KNN, and Logistic Regression.
        """
        algorithms = ["random_forest", "svm", "knn", "logistic_regression"]
        results = []

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        for algo in algorithms:
            model = self.build_classifier(algo)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append({
                "algorithm": algo.replace("_", " ").title(),
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1 * 100, 2)
            })

        return sorted(results, key=lambda x: x["accuracy"], reverse=True)
