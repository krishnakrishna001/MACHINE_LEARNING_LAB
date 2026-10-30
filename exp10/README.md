# Skin Disease Detection Using Machine Learning
### An All-in-One "One Frame" Interactive Web Diagnostic Application

---

## 📌 Project Overview
**Skin Disease Detection Using Machine Learning** is an end-to-end computer-aided diagnostic (CAD) system for dermatology. It incorporates all 5 core stages of the machine learning lifecycle inside a **single unified frame ("One-Frame") web application**:

1. **Load Input Image:** Upload raw dermoscopy or clinical skin photos, or select from pre-loaded clinical case studies.
2. **Preprocessor:** Executes the dermatological **Dull Razor hair-removal algorithm**, Gaussian spatial denoising, **CLAHE** (Contrast Limited Adaptive Histogram Equalization), and automated **Otsu lesion segmentation** with contour bounding overlays.
3. **Feature Extraction:** Extracts a comprehensive **42-dimensional quantitative feature vector** across 4 clinical domains (Color Moments in RGB & HSV, GLCM Haralick Textures, ABCD Morphological/Shape Metrics, and Local Binary Pattern Micro-texture).
4. **Model Training:** Interactive in-browser training engine supporting **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **Logistic Regression** with real-time hyperparameter tuning and train/test ratio control.
5. **Performance Evaluation:** Live evaluation dashboard rendering **Accuracy, Precision, Recall, F1-Score, 5-Fold Stratified Cross-Validation, Confusion Matrix heatmaps, Multi-Class ROC Curves, Classification Reports, and Cross-Algorithm Benchmarks**.
6. **Clinical Diagnostic Prediction:** Generates pathology classification, confidence scores, disease severity, clinical symptoms, and medical recommendations.

Supported skin conditions:
* **Melanoma (MEL)** – High-risk malignant skin cancer
* **Eczema / Atopic Dermatitis (ECZ)** – Chronic inflammatory pruritic condition
* **Psoriasis (PSO)** – Scaly autoimmune plaque formation
* **Benign Keratosis (BKL)** – Non-cancerous seborrheic keratosis or nevus
* **Acne Vulgaris (ACN)** – Inflammatory papules and pustules
* **Basal Cell Carcinoma (BCC)** – Locally invasive translucent cancer

---

## 📂 Project Directory Structure

```
d:\krishna\
│
├── app.py                      # Flask Backend & API routes for the One-Frame Dashboard
├── preprocessor.py             # Image Preprocessing: Dull Razor, CLAHE, Denoising, Otsu Segmentation
├── feature_extraction.py       # 42-Feature Extractor: Color Moments, GLCM, ABCD Shape, LBP
├── train_model.py              # ML Engine: Random Forest, SVM, KNN, Evaluation Metrics, Plots
├── dataset_generator.py        # Generates balanced dataset & initial pre-trained models
├── test_system.py              # Automated Unit Test Suite verifying all 5 stages
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation, compilation & running guide
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern medical-tech dark-mode UI stylesheet
│   ├── js/
│   │   └── main.js             # Interactive client-side logic, AJAX processing, Chart.js visuals
│   ├── samples/                # Sample test images for all 6 conditions
│   │   ├── melanoma.jpg
│   │   ├── eczema.jpg
│   │   ├── psoriasis.jpg
│   │   ├── benign_keratosis.jpg
│   │   ├── acne_vulgaris.jpg
│   │   └── basal_cell_carcinoma.jpg
│   └── uploads/                # User uploaded images
│
├── templates/
│   └── index.html              # "One-Frame" Single-Page Responsive Dashboard
│
├── data/
│   ├── X_skin_features.npy     # 360 Samples × 42 Features matrix
│   └── y_skin_labels.npy       # Ground truth diagnostic labels
│
└── models/
    ├── skin_random_forest_model.joblib
    ├── skin_svm_model.joblib
    ├── skin_disease_scaler.joblib
    └── model_metadata.json
```

---

## ⚙️ How to Compile, Setup, and Run the Project

### Prerequisites
* **Operating System:** Windows, macOS, or Linux
* **Python Version:** Python 3.10, 3.11, or 3.12 installed
* **Browser:** Chrome, Firefox, Edge, or Safari

---

### Step 1: Open Terminal / PowerShell
Open your command prompt or PowerShell and navigate to the project directory:
```powershell
cd d:\krishna
```

---

### Step 2: Install Required Python Libraries
Run the following command to install the required packages:
```powershell
pip install -r requirements.txt
```
*Dependencies installed:*
* `flask` - Web application framework
* `opencv-python` - Computer vision and morphological filters
* `numpy` & `scipy` - Scientific computing, statistics, matrix math
* `scikit-learn` - Machine learning algorithms and performance evaluation
* `matplotlib` - Headless visualization for confusion matrix & ROC curves
* `pillow` - Image processing support
* `joblib` - Model persistence

---

### Step 3: Verify or Regenerate the Dataset (Optional)
The dataset (`data/`) and pre-trained models (`models/`) are already generated and included. If you wish to regenerate or retrain fresh models:
```powershell
python dataset_generator.py
```
*Output will display:*
```
[INFO] Generating dermatological dataset and sample images...
[SUCCESS] Dataset generated: 360 samples with 42 features across 6 classes.
[INFO] Training default Random Forest Classifier...
[SUCCESS] Random Forest trained with Accuracy: 100.0%
[INFO] Training default Support Vector Machine (SVM)...
[SUCCESS] SVM trained with Accuracy: 100.0%
```

---

### Step 4: Run the Automated Test Suite
To verify that all 5 pipeline stages and web routes are functioning perfectly on your machine:
```powershell
python test_system.py
```
*Expected test output:*
```
[TEST 1] Testing Image Preprocessing Pipeline...
  --> Preprocessing PASSED (All 6 intermediate stages generated).
[TEST 2] Testing 42-Feature Extraction...
  --> Feature Extraction PASSED (Extracted 42 features across 4 clinical domains).
[TEST 3] Testing Flask Web Application Endpoints...
  --> GET / (One-Frame Dashboard) PASSED.
  --> POST /api/process-image PASSED.
  --> POST /api/predict PASSED (Predicted: Melanoma, Confidence: 90.67%).
  --> GET /api/benchmark PASSED (4 models compared).
----------------------------------------------------------------------
Ran 3 tests in 1.2s
OK
```

---

### Step 5: Compile and Run the Web Application
Start the Flask local development server:
```powershell
python app.py
```
*Terminal output:*
```
[SERVER] Skin Disease Detection System starting on http://127.0.0.1:5000 ...
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

### Step 6: Open the Application in Your Browser
Open your browser and navigate to:
```
http://localhost:5000
```
or
```
http://127.0.0.1:5000
```

---

## 🖥️ Using the "One Frame" Dashboard

Everything is organized within **one responsive dashboard frame**:

1. **Stage 1 (Load Image):**
   * Drag and drop any image into the dashed upload area, or click one of the 6 instant clinical sample buttons (**Melanoma**, **Eczema**, **Psoriasis**, **Benign Keratosis**, **Acne**, or **Basal Cell Carcinoma**).

2. **Stage 2 (Preprocessor Inspection):**
   * Use the stage selector tabs above the viewer to toggle between:
     * **Original:** Standardized RGB image.
     * **Dull Razor:** Hair-removed inpainting result.
     * **Denoised:** Gaussian smoothed image.
     * **CLAHE:** Contrast-enhanced L-channel view.
     * **Otsu Mask:** Binary segmented lesion mask.
     * **Lesion Contour:** Green outline and orange bounding box identifying the pathological zone.

3. **Stage 3 (Feature Extraction):**
   * View the 42 extracted numerical features in the interactive grid.
   * Filter by category: **Color Moments (18)**, **GLCM Texture (6)**, **Shape & ABCD (8)**, or **LBP Micro-texture (10)**.
   * Inspect the **Radar Chart** representing the lesion's multi-dimensional morphological fingerprint.

4. **Stage 4 (Model Training):**
   * Choose an algorithm: **Random Forest**, **SVM**, **KNN**, or **Logistic Regression**.
   * Adjust hyperparameters (Trees, Max Depth, Kernel, C, k) and Train/Test Split slider.
   * Click **"Train Model & Evaluate Performance"** to re-fit the model on-the-fly and observe real-time logs.

5. **Stage 5 (Performance Evaluation & Diagnostics):**
   * Review test metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
   * Inspect the **Confusion Matrix Heatmap** and **Multi-Class ROC Curves**.
   * View the detailed **Classification Report** table.
   * Click **"⚡ Benchmark All"** in the top navigation bar to compare all 4 algorithms on the leaderboard.
   * Read the **Clinical Diagnostic Prediction** card for severity classification and patient guidance.

---

## 🔬 Mathematical & Algorithmic Background

### 1. Dull Razor Hair Removal
Thin dark hair filaments are detected using a morphological Black-Hat transform with a structuring element $B$:
$$\text{BlackHat}(I) = (I \bullet B) - I$$
Pixels above threshold are identified as hair artifacts and inpainted using the Telea Fast Marching algorithm from surrounding healthy dermis pixels.

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
Operates on the Luminance ($L^*$) channel of the CIELAB color space. Prevents noise amplification by clipping histograms at limit $\beta=2.5$ before calculating the cumulative distribution function (CDF).

### 3. Otsu's Automated Thresholding
Finds the threshold $t^*$ that minimizes intra-class variance $\sigma_w^2(t)$:
$$\sigma_w^2(t) = \omega_0(t) \sigma_0^2(t) + \omega_1(t) \sigma_1^2(t)$$

### 4. GLCM (Gray-Level Co-occurrence Matrix) Haralick Metrics
For normalized co-occurrence matrix $P(i, j)$:
* **Contrast:** $\sum_{i,j} |i - j|^2 P(i,j)$
* **Homogeneity:** $\sum_{i,j} \frac{P(i,j)}{1 + (i - j)^2}$
* **Energy (ASM):** $\sum_{i,j} P(i,j)^2$
* **Entropy:** $-\sum_{i,j} P(i,j) \log_2(P(i,j) + \epsilon)$

### 5. ABCD Rule Metrics
* **Asymmetry Index ($A$):** Ratio of non-overlapping pixels across horizontal and vertical mirror axes:
  $$A_{idx} = \frac{|A_{\text{left}} - A_{\text{right}}| + |A_{\text{top}} - A_{\text{bottom}}|}{2 \cdot A_{\text{total}}}$$
* **Border Circularity / Compactness ($B$):**
  $$C = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2} \quad (1.0 = \text{perfect circle})$$

---

## 🏥 Clinical Disclaimer
*This system is intended for educational, academic, and decision-support triage research. Formal medical diagnoses should always be confirmed by a licensed board-certified dermatologist.*

