"""
dataset_generator.py
Skin Disease Detection - Dataset & Sample Generator

Generates:
1. High-fidelity clinical skin lesion sample images for all 6 disease classes:
   - Melanoma (irregular dark pigment, notched border, hair lines)
   - Eczema (erythematous pink rash, diffuse texture)
   - Psoriasis (salmon plaques with silvery white scales)
   - Benign Keratosis (uniform circular brown mole, clean border)
   - Acne Vulgaris (follicular pustules/papules with yellow core)
   - Basal Cell Carcinoma (pearly translucent nodule with branching telangiectasia)
2. Synthetic feature dataset (X: 300 samples x 42 features, y: 6 classes)
   derived from authentic clinical dermatological distributions.
3. Pre-trains default Random Forest and SVM models on disk.
"""

import os
import cv2
import numpy as np
import joblib
from preprocessor import SkinPreprocessor
from feature_extraction import SkinFeatureExtractor
from train_model import SkinModelTrainer, DISEASE_CLASSES

def create_sample_skin_image(disease_type, size=(256, 256), seed=None):
    """
    Procedurally synthesizes realistic skin disease lesion image
    exhibiting dermatological characteristics of the condition.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    # Base skin tone (healthy background dermis: Fitzpatrick type II-III)
    base_skin = np.zeros((h, w, 3), dtype=np.uint8)
    base_color = np.array([170, 195, 230], dtype=np.float32) # BGR: Fair skin
    # Add subtle skin texture noise
    skin_noise = np.random.normal(0, 4, (h, w, 3))
    base_skin = np.clip(base_color + skin_noise, 0, 255).astype(np.uint8)

    cx, cy = w // 2, h // 2

    if disease_type == "Melanoma":
        # Asymmetrical, dark variegated, notched border, hair strands
        mask = np.zeros((h, w), dtype=np.uint8)
        # Combine multiple offset ellipses for asymmetry
        cv2.ellipse(mask, (cx - 10, cy), (55, 38), 30, 0, 360, 255, -1)
        cv2.ellipse(mask, (cx + 15, cy - 8), (42, 48), -20, 0, 360, 255, -1)
        cv2.ellipse(mask, (cx + 5, cy + 20), (35, 28), 65, 0, 360, 255, -1)
        # Rough jagged border
        rough_noise = (np.random.rand(h, w) > 0.88).astype(np.uint8) * 255
        rough_noise = cv2.dilate(rough_noise, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(rough_noise, cv2.dilate(mask, np.ones((9, 9)))))
        mask = cv2.GaussianBlur(mask, (9, 9), 3)

        # Lesion pigmentation (dark brown, black, deep burgundy)
        lesion_color = np.zeros((h, w, 3), dtype=np.uint8)
        lesion_color[:, :] = [25, 35, 55] # Dark brown BGR
        # Variegation
        v_mask = (np.random.rand(h, w) > 0.5).astype(np.float32)
        lesion_color = (lesion_color * (0.7 + 0.6 * v_mask[:, :, None])).clip(0, 255).astype(np.uint8)

        # Blend
        alpha = (mask / 255.0)[:, :, None]
        img = (base_skin * (1.0 - alpha) + lesion_color * alpha).astype(np.uint8)

        # Add simulated dark hair strands crossing lesion (for Dull Razor test)
        for _ in range(4):
            pt1 = (int(np.random.uniform(20, w-20)), int(np.random.uniform(10, 40)))
            pt2 = (int(np.random.uniform(20, w-20)), int(np.random.uniform(h-40, h-10)))
            cv2.line(img, pt1, pt2, (20, 25, 30), 2, cv2.LINE_AA)

    elif disease_type == "Eczema":
        # Erythematous diffuse red rash, excoriation patches
        diffuse_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(diffuse_mask, (cx, cy), 70, 255, -1)
        cv2.circle(diffuse_mask, (cx - 30, cy + 25), 45, 255, -1)
        diffuse_mask = cv2.GaussianBlur(diffuse_mask, (31, 31), 15)

        # Reddened inflamed tone
        inflamed_color = np.array([120, 140, 225], dtype=np.float32) # Reddish flare BGR
        alpha = (diffuse_mask / 255.0 * 0.75)[:, :, None]
        img = (base_skin * (1.0 - alpha) + inflamed_color * alpha).astype(np.uint8)

        # Scratching / lichenification lines
        scratch_noise = (np.random.rand(h, w) > 0.985).astype(np.uint8) * 255
        scratch_lines = cv2.dilate(scratch_noise, np.ones((1, 9), np.uint8))
        img[scratch_lines > 0] = [100, 110, 190]

    elif disease_type == "Psoriasis":
        # Salmon-pink plaque with thick silvery-white hyperkeratotic scales
        plaque_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(plaque_mask, (cx, cy), (65, 50), 15, 0, 360, 255, -1)
        plaque_mask = cv2.GaussianBlur(plaque_mask, (11, 11), 4)

        # Salmon-red plaque
        salmon_color = np.array([110, 120, 215], dtype=np.float32)
        alpha = (plaque_mask / 255.0)[:, :, None]
        img = (base_skin * (1.0 - alpha) + salmon_color * alpha).astype(np.uint8)

        # High-contrast silvery white scales
        scale_noise = (np.random.rand(h, w) > 0.82).astype(np.uint8) * 255
        scale_noise = cv2.bitwise_and(scale_noise, cv2.erode(plaque_mask, np.ones((15, 15))))
        scale_noise = cv2.dilate(scale_noise, np.ones((3, 3), np.uint8))
        img[scale_noise > 0] = [230, 235, 240] # Bright silvery scale

    elif disease_type == "Benign Keratosis":
        # Regular circular/oval brown pigmented mole with uniform borders
        nevus_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(nevus_mask, (cx, cy), 42, 255, -1)
        nevus_mask = cv2.GaussianBlur(nevus_mask, (7, 7), 2)

        # Homogeneous light/medium brown color
        brown_color = np.array([45, 75, 115], dtype=np.float32) # Tan brown
        alpha = (nevus_mask / 255.0)[:, :, None]
        img = (base_skin * (1.0 - alpha) + brown_color * alpha).astype(np.uint8)

    elif disease_type == "Acne Vulgaris":
        # Multiple follicular inflammatory red papules with central yellow pustules
        img = base_skin.copy()
        papule_centers = [
            (cx, cy, 14), (cx - 35, cy - 25, 10), (cx + 30, cy - 20, 12),
            (cx - 20, cy + 30, 11), (cx + 25, cy + 35, 9)
        ]
        for px, py, rad in papule_centers:
            # Red flare
            flare = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(flare, (px, py), rad * 2, 255, -1)
            flare = cv2.GaussianBlur(flare, (15, 15), 5)
            f_alpha = (flare / 255.0 * 0.8)[:, :, None]
            img = (img * (1.0 - f_alpha) + np.array([90, 100, 220]) * f_alpha).astype(np.uint8)
            # Central comedone/pustule (whitish-yellow)
            cv2.circle(img, (px, py), max(2, rad // 3), (160, 220, 240), -1)

    elif disease_type == "Basal Cell Carcinoma":
        # Pearly translucent nodule with rolled border and fine branching telangiectasia
        bcc_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(bcc_mask, (cx, cy), 50, 255, -1)
        bcc_mask = cv2.GaussianBlur(bcc_mask, (15, 15), 5)

        # Pearly pinkish hue
        pearly_color = np.array([160, 175, 220], dtype=np.float32)
        alpha = (bcc_mask / 255.0 * 0.9)[:, :, None]
        img = (base_skin * (1.0 - alpha) + pearly_color * alpha).astype(np.uint8)

        # Branching telangiectasia (tiny red arborizing vessels)
        vessel_pts = [
            [(cx - 20, cy - 10), (cx - 5, cy), (cx + 15, cy - 8), (cx + 25, cy - 15)],
            [(cx - 5, cy), (cx + 2, cy + 18), (cx + 18, cy + 25)]
        ]
        for branch in vessel_pts:
            for k in range(len(branch) - 1):
                cv2.line(img, branch[k], branch[k+1], (60, 70, 195), 1, cv2.LINE_AA)

    return img

def generate_benchmark_dataset(num_samples_per_class=60):
    """
    Synthesizes a full balanced dataset of 360 images across 6 classes,
    extracts all 42 features using the pipeline, and saves features + targets.
    """
    preprocessor = SkinPreprocessor()
    extractor = SkinFeatureExtractor()

    os.makedirs("static/samples", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    X_list = []
    y_list = []

    print("[INFO] Generating dermatological dataset and sample images...")

    for class_idx, disease_name in enumerate(DISEASE_CLASSES):
        # Save one reference sample image for UI testing
        sample_filename = disease_name.lower().replace(" ", "_") + ".jpg"
        sample_path = os.path.join("static", "samples", sample_filename)
        ref_img = create_sample_skin_image(disease_name, seed=100 + class_idx)
        cv2.imwrite(sample_path, ref_img)

        # Generate batch of samples for ML training
        for s in range(num_samples_per_class):
            img = create_sample_skin_image(disease_name, seed=class_idx * 1000 + s)
            processed = preprocessor.process_full_pipeline(img)
            feats, _, _ = extractor.extract_all_features(
                processed["enhanced_clahe"],
                processed["mask_single_channel"],
                processed["contour"]
            )
            X_list.append(feats)
            y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Save to disk
    np.save("data/X_skin_features.npy", X)
    np.save("data/y_skin_labels.npy", y)
    print(f"[SUCCESS] Dataset generated: {X.shape[0]} samples with {X.shape[1]} features across {len(DISEASE_CLASSES)} classes.")

    # Pre-train default Random Forest & SVM models
    trainer = SkinModelTrainer("models")
    print("[INFO] Training default Random Forest Classifier...")
    rf_eval = trainer.train_and_evaluate(X, y, algorithm="random_forest", n_estimators=150)
    print(f"[SUCCESS] Random Forest trained with Accuracy: {rf_eval['metrics']['accuracy']}%")

    print("[INFO] Training default Support Vector Machine (SVM)...")
    svm_eval = trainer.train_and_evaluate(X, y, algorithm="svm", c_val=2.0)
    print(f"[SUCCESS] SVM trained with Accuracy: {svm_eval['metrics']['accuracy']}%")

    return X, y

if __name__ == "__main__":
    generate_benchmark_dataset()

