/**
 * main.js - DermAI Skin Disease Detection Frontend Logic
 * Manages full lifecycle: Upload -> Preprocess -> Feature Extraction -> Training -> Evaluation -> Diagnosis
 */

let currentProcessedImages = {};
let currentFeatures = null;
let currentFeatureBreakdown = null;
let featureRadarChartInstance = null;
let probabilityChartInstance = null;

const stageDescriptions = {
    original: "<strong class='text-cyan-400'>Original Image:</strong> Standardized 256x256 dermatological image.",
    hair_removed: "<strong class='text-cyan-400'>Dull Razor Filter:</strong> Morphological black hat transform isolates hair artifacts; inpainting restores lesion dermis.",
    denoised: "<strong class='text-cyan-400'>Denoised:</strong> 5x5 Gaussian spatial smoothing eliminates camera sensor noise.",
    enhanced_clahe: "<strong class='text-cyan-400'>CLAHE:</strong> Contrast Limited Adaptive Histogram Equalization in LAB space reveals pigment patterns.",
    binary_mask: "<strong class='text-cyan-400'>Otsu Segmentation:</strong> Bimodal automated thresholding isolates pathological lesion from healthy skin.",
    contour_overlay: "<strong class='text-cyan-400'>Lesion Contour:</strong> Green boundary perimeter and orange minimum bounding box around lesion."
};

document.addEventListener("DOMContentLoaded", () => {
    setupUploadHandlers();
    initCharts();
    toggleAlgorithmHyperparams();
    // Auto-load melanoma sample on initial page load to immediately populate the frame!
    loadSampleImage('melanoma.jpg', 'Melanoma (Malignant)');
});

// Setup drag and drop / file selection
function setupUploadHandlers() {
    const dropZone = document.getElementById("dropZone");
    const imageInput = document.getElementById("imageInput");

    dropZone.addEventListener("click", () => imageInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add("border-cyan-400"), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove("border-cyan-400"), false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    imageInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });
}

function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("image", file);

    logToConsole(`[UPLOAD] Uploading ${file.name} (${(file.size / 1024).toFixed(1)} KB)...`);
    setPipelineStep(1);

    fetch("/api/process-image", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => handleProcessResponse(data, file.name))
    .catch(err => {
        logToConsole(`[ERROR] Image processing failed: ${err.message}`, "text-rose-400");
    });
}

function loadSampleImage(sampleFile, sampleName) {
    logToConsole(`[LOAD] Loading sample clinical case: ${sampleName}...`);
    setPipelineStep(1);

    fetch("/api/process-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_file: sampleFile })
    })
    .then(res => res.json())
    .then(data => handleProcessResponse(data, sampleFile))
    .catch(err => {
        logToConsole(`[ERROR] Sample loading failed: ${err.message}`, "text-rose-400");
    });
}

function handleProcessResponse(data, filename) {
    if (!data.success) {
        alert("Processing Error: " + data.error);
        logToConsole(`[ERROR] ${data.error}`, "text-rose-400");
        return;
    }

    currentProcessedImages = data.images;
    currentFeatures = data.features_summary.feature_values;
    currentFeatureBreakdown = data.features_summary.breakdown;

    // Update specs box
    document.getElementById("imageSpecsBox").classList.remove("hidden");
    document.getElementById("specFileName").textContent = filename;

    // Update intermediate thumbnails
    for (const [key, base64] of Object.entries(data.images)) {
        const thumb = document.getElementById(`thumb_${key}`);
        if (thumb) thumb.src = base64;
    }

    // Default display to contour overlay
    switchPreprocessView("contour_overlay");

    // Populate Feature Cards & Radar chart
    renderFeatureCards(data.features_summary.feature_names, data.features_summary.feature_values);
    updateRadarChart(data.features_summary.breakdown);

    setPipelineStep(3);
    logToConsole(`[SUCCESS] Preprocessing completed & 42 features extracted successfully!`, "text-emerald-400");

    // Run prediction automatically for seamless user experience
    runInference();
}

// Switch between intermediate preprocessor images
function switchPreprocessView(stageKey) {
    if (!currentProcessedImages[stageKey]) return;

    // Highlight active tab
    document.querySelectorAll(".prep-tab").forEach(tab => tab.classList.remove("active"));
    const activeTab = document.getElementById(`tab_${stageKey}`);
    if (activeTab) activeTab.classList.add("active");

    const mainImg = document.getElementById("mainPreprocessDisplay");
    const placeholder = document.getElementById("preprocessPlaceholder");
    const desc = document.getElementById("stageDescription");

    mainImg.src = currentProcessedImages[stageKey];
    mainImg.classList.remove("hidden");
    placeholder.classList.add("hidden");

    desc.innerHTML = stageDescriptions[stageKey] || "";
    desc.classList.remove("hidden");
}

// Render 42 feature cards in grid
function renderFeatureCards(names, values, categoryFilter = "all") {
    const container = document.getElementById("featureGridContainer");
    container.innerHTML = "";

    let displayedCount = 0;

    names.forEach((name, idx) => {
        const val = values[idx];
        let category = "other";

        if (idx < 18) category = "color";
        else if (idx < 24) category = "texture";
        else if (idx < 32) category = "shape";
        else category = "lbp";

        if (categoryFilter !== "all" && category !== categoryFilter) {
            return;
        }

        displayedCount++;
        const card = document.createElement("div");
        card.className = "p-2 rounded-lg bg-dark-900 border border-slate-800 text-xs flex flex-col justify-between hover:border-slate-700 transition";
        card.innerHTML = `
            <span class="text-[10px] text-slate-400 font-mono truncate" title="${name}">${name}</span>
            <span class="text-sm font-bold font-mono text-cyan-300 mt-1">${typeof val === 'number' ? val.toFixed(3) : val}</span>
        `;
        container.appendChild(card);
    });

    if (displayedCount === 0) {
        container.innerHTML = `<div class="col-span-full text-center py-8 text-slate-500 text-xs">No features in this category.</div>`;
    }
}

// Filter features by tab
function switchFeatureTab(cat) {
    document.querySelectorAll(".feat-tab").forEach(t => t.classList.remove("active"));
    const tab = document.getElementById(`ftab_${cat}`);
    if (tab) tab.classList.add("active");

    if (currentFeatures) {
        fromFeatureExtractorNames = [
            "R_Mean", "G_Mean", "B_Mean", "R_Std", "G_Std", "B_Std", "R_Skew", "G_Skew", "B_Skew",
            "H_Mean", "S_Mean", "V_Mean", "H_Std", "S_Std", "V_Std", "H_Skew", "S_Skew", "V_Skew",
            "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity", "GLCM_Energy", "GLCM_Correlation", "GLCM_Entropy",
            "Asymmetry_Index", "Border_Circularity", "Solidity", "Aspect_Ratio", "Extent", "Eccentricity", "Hu_Moment_1", "Hu_Moment_2",
            "LBP_Bin_0", "LBP_Bin_1", "LBP_Bin_2", "LBP_Bin_3", "LBP_Bin_4", "LBP_Bin_5", "LBP_Bin_6", "LBP_Bin_7", "LBP_Bin_8", "LBP_NonUniform"
        ];
        renderFeatureCards(fromFeatureExtractorNames, currentFeatures, cat);
    }
}

// Initialize Chart.js instances
function initCharts() {
    // Feature Radar Chart
    const ctxRadar = document.getElementById("featureRadarChart").getContext("2d");
    featureRadarChartInstance = new Chart(ctxRadar, {
        type: "radar",
        data: {
            labels: ["Color Var", "GLCM Contrast", "Homogeneity", "Asymmetry", "Circularity", "Solidity", "Entropy", "Micro-texture"],
            datasets: [{
                label: "Lesion Feature Signature",
                data: [0.5, 0.4, 0.7, 0.3, 0.6, 0.8, 0.5, 0.4],
                backgroundColor: "rgba(139, 92, 246, 0.25)",
                borderColor: "#a855f7",
                borderWidth: 2,
                pointBackgroundColor: "#c084fc",
                pointBorderColor: "#ffffff",
                pointRadius: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "rgba(51, 65, 85, 0.4)" },
                    grid: { color: "rgba(51, 65, 85, 0.3)" },
                    pointLabels: { color: "#94a3b8", font: { size: 9 } },
                    ticks: { display: false, maxTicksLimit: 4 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // Probability Distribution Bar Chart
    const ctxProb = document.getElementById("probabilityChart").getContext("2d");
    probabilityChartInstance = new Chart(ctxProb, {
        type: "bar",
        data: {
            labels: ["Melanoma", "Eczema", "Psoriasis", "Benign Ker.", "Acne", "BCC"],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    "#f43f5e", "#38bdf8", "#fbbf24", "#4ade80", "#a855f7", "#ec4899"
                ],
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    max: 100,
                    grid: { color: "rgba(51, 65, 85, 0.2)" },
                    ticks: { color: "#64748b", font: { size: 9 }, callback: v => v + '%' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#cbd5e1", font: { size: 9 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: ctx => ` ${ctx.raw}% probability` }
                }
            }
        }
    });
}

function updateRadarChart(breakdown) {
    if (!featureRadarChartInstance || !breakdown) return;

    // Compute normalized representative values for radar axes
    const colorVar = Math.min(1.0, (breakdown["Color Moments (RGB & HSV)"]["R_Std"] || 20) / 60.0);
    const contrast = Math.min(1.0, (breakdown["GLCM Texture Analysis"]["GLCM_Contrast"] || 10) / 40.0);
    const homogeneity = Math.min(1.0, breakdown["GLCM Texture Analysis"]["GLCM_Homogeneity"] || 0.6);
    const asymmetry = Math.min(1.0, (breakdown["Shape & ABCD Metrics"]["Asymmetry_Index"] || 0.2) * 2.5);
    const circularity = Math.min(1.0, breakdown["Shape & ABCD Metrics"]["Border_Circularity"] || 0.5);
    const solidity = Math.min(1.0, breakdown["Shape & ABCD Metrics"]["Solidity"] || 0.8);
    const entropy = Math.min(1.0, (breakdown["GLCM Texture Analysis"]["GLCM_Entropy"] || 3.0) / 6.0);
    const micro = Math.min(1.0, (breakdown["LBP Micro-texture Histogram"]["LBP_Bin_0"] || 0.2) * 3.0);

    featureRadarChartInstance.data.datasets[0].data = [
        colorVar, contrast, homogeneity, asymmetry, circularity, solidity, entropy, micro
    ];
    featureRadarChartInstance.update();
}

// Toggle hyperparameter inputs when selecting algorithm
function toggleAlgorithmHyperparams() {
    const algo = document.getElementById("trainAlgorithm").value;
    document.getElementById("paramGroupRF").classList.add("hidden");
    document.getElementById("paramGroupSVM").classList.add("hidden");
    document.getElementById("paramGroupKNN").classList.add("hidden");

    if (algo === "random_forest") {
        document.getElementById("paramGroupRF").classList.remove("hidden");
    } else if (algo === "svm") {
        document.getElementById("paramGroupSVM").classList.remove("hidden");
    } else if (algo === "knn") {
        document.getElementById("paramGroupKNN").classList.remove("hidden");
    }
}

function updateSplitDisplay(val) {
    const trainVal = 100 - parseInt(val);
    document.getElementById("splitRatioDisplay").textContent = `${trainVal}% / ${val}%`;
}

// Train Active Machine Learning Model
function trainActiveModel() {
    const algo = document.getElementById("trainAlgorithm").value;
    const testSplit = parseInt(document.getElementById("splitSlider").value) / 100.0;
    const trainBtn = document.getElementById("trainBtn");

    trainBtn.disabled = true;
    trainBtn.classList.add("opacity-50");
    document.getElementById("trainingStatusTag").textContent = "Training in progress...";
    document.getElementById("trainingStatusTag").className = "text-amber-400 animate-pulse text-[10px]";

    const payload = {
        algorithm: algo,
        test_size: testSplit
    };

    if (algo === "random_forest") {
        payload.n_estimators = document.getElementById("paramNEstimators").value;
        payload.max_depth = document.getElementById("paramMaxDepth").value;
    } else if (algo === "svm") {
        payload.kernel = document.getElementById("paramSVMKernel").value;
        payload.c_val = document.getElementById("paramSVMC").value;
    } else if (algo === "knn") {
        payload.n_neighbors = document.getElementById("paramKNNNeighbors").value;
    }

    logToConsole(`[TRAIN] Initiating training for ${algo.toUpperCase()} on 360 sample vectors...`);
    setPipelineStep(4);

    fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        trainBtn.disabled = false;
        trainBtn.classList.remove("opacity-50");
        document.getElementById("trainingStatusTag").textContent = "Completed";
        document.getElementById("trainingStatusTag").className = "text-emerald-400 text-[10px]";

        if (!data.success) {
            logToConsole(`[ERROR] Training failed: ${data.error}`, "text-rose-400");
            alert("Training Error: " + data.error);
            return;
        }

        // Update KPIs
        document.getElementById("evalAccuracy").textContent = `${data.metrics.accuracy}%`;
        document.getElementById("evalPrecision").textContent = `${data.metrics.precision}%`;
        document.getElementById("evalRecall").textContent = `${data.metrics.recall}%`;
        document.getElementById("evalF1").textContent = `${data.metrics.f1_score}%`;
        document.getElementById("cvScoreDisplay").textContent = `${data.metrics.cv_mean}% (±${data.metrics.cv_std}%)`;
        document.getElementById("testSamplesCount").textContent = `${data.metrics.test_samples} test cases`;

        // Update Header Badges
        document.getElementById("headerActiveModel").textContent = data.algorithm.replace("_", " ").toUpperCase();
        document.getElementById("headerAccuracy").textContent = `${data.metrics.accuracy}%`;

        // Update Plots
        if (data.confusion_matrix_img) {
            const cmImg = document.getElementById("cmPlotImg");
            cmImg.src = data.confusion_matrix_img;
            cmImg.classList.remove("hidden");
            document.getElementById("cmPlotPlaceholder").classList.add("hidden");
        }

        if (data.roc_curve_img) {
            const rocImg = document.getElementById("rocPlotImg");
            rocImg.src = data.roc_curve_img;
            rocImg.classList.remove("hidden");
            document.getElementById("rocPlotPlaceholder").classList.add("hidden");
        }

        // Render Classification Report Table
        renderClassificationReportTable(data.classification_report, data.class_names);

        logToConsole(`[SUCCESS] Training finished! Test Accuracy: ${data.metrics.accuracy}% | 5-Fold CV: ${data.metrics.cv_mean}%`, "text-emerald-400");
        setPipelineStep(5);

        // Re-run diagnostic inference with new model if image exists
        if (currentFeatures) {
            runInference();
        }
    })
    .catch(err => {
        trainBtn.disabled = false;
        trainBtn.classList.remove("opacity-50");
        logToConsole(`[ERROR] Training request error: ${err.message}`, "text-rose-400");
    });
}

function renderClassificationReportTable(report, classNames) {
    const tbody = document.getElementById("reportTableBody");
    tbody.innerHTML = "";

    classNames.forEach(cName => {
        const stats = report[cName];
        if (!stats) return;
        const row = document.createElement("tr");
        row.className = "hover:bg-dark-900/40";
        row.innerHTML = `
            <td class="py-1.5 px-3 font-medium text-white">${cName}</td>
            <td class="py-1.5 px-3 font-mono text-cyan-400">${(stats.precision * 100).toFixed(1)}%</td>
            <td class="py-1.5 px-3 font-mono text-purple-400">${(stats.recall * 100).toFixed(1)}%</td>
            <td class="py-1.5 px-3 font-mono text-amber-400">${(stats['f1-score'] * 100).toFixed(1)}%</td>
            <td class="py-1.5 px-3 font-mono text-slate-400">${stats.support}</td>
        `;
        tbody.appendChild(row);
    });
}

// Switch between Performance Evaluation tabs
function switchEvalTab(tabKey) {
    document.querySelectorAll(".eval-tab").forEach(t => t.classList.remove("active"));
    const activeTab = document.getElementById(`etab_${tabKey}`);
    if (activeTab) activeTab.classList.add("active");

    document.getElementById("evalView_cm").classList.add("hidden");
    document.getElementById("evalView_roc").classList.add("hidden");
    document.getElementById("evalView_report").classList.add("hidden");
    document.getElementById("evalView_benchmark").classList.add("hidden");

    const targetView = document.getElementById(`evalView_${tabKey}`);
    if (targetView) targetView.classList.remove("hidden");
}

// Benchmark All 4 Algorithms
function runBenchmark() {
    logToConsole(`[BENCHMARK] Benchmarking Random Forest, SVM, KNN, and Logistic Regression...`);
    switchEvalTab("benchmark");

    const tbody = document.getElementById("benchmarkTableBody");
    tbody.innerHTML = `<tr><td colspan="5" class="py-6 text-center text-cyan-400 animate-pulse">Running cross-algorithm benchmark on dataset...</td></tr>`;

    fetch("/api/benchmark")
    .then(res => res.json())
    .then(data => {
        if (!data.success) {
            tbody.innerHTML = `<tr><td colspan="5" class="py-4 text-center text-rose-400">Benchmark error: ${data.error}</td></tr>`;
            return;
        }

        tbody.innerHTML = "";
        data.benchmark.forEach((item, idx) => {
            const row = document.createElement("tr");
            row.className = idx === 0 ? "bg-cyan-950/20 font-bold" : "hover:bg-dark-900/40";
            row.innerHTML = `
                <td class="py-2 px-3 text-white flex items-center gap-1.5">
                    ${idx === 0 ? '<span class="text-amber-400">★</span>' : ''}
                    <span>${item.algorithm}</span>
                </td>
                <td class="py-2 px-3 font-mono text-cyan-400">${item.accuracy}%</td>
                <td class="py-2 px-3 font-mono text-emerald-400">${item.precision}%</td>
                <td class="py-2 px-3 font-mono text-purple-400">${item.recall}%</td>
                <td class="py-2 px-3 font-mono text-amber-400">${item.f1_score}%</td>
            `;
            tbody.appendChild(row);
        });

        logToConsole(`[SUCCESS] Benchmark completed. Top model: ${data.benchmark[0].algorithm} (${data.benchmark[0].accuracy}% acc)`, "text-emerald-400");
    })
    .catch(err => {
        logToConsole(`[ERROR] Benchmark error: ${err.message}`, "text-rose-400");
    });
}

// Run Diagnostic Inference
function runInference() {
    const btn = document.getElementById("predictBtn");
    btn.disabled = true;

    fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    })
    .then(res => res.json())
    .then(data => {
        btn.disabled = false;
        if (!data.success) {
            logToConsole(`[WARNING] Prediction: ${data.error}`, "text-amber-400");
            return;
        }

        // Show result box
        document.getElementById("diagnosisPlaceholder").classList.add("hidden");
        document.getElementById("diagnosisResult").classList.remove("hidden");

        // Disease name & confidence
        document.getElementById("diagDiseaseName").textContent = data.disease_name;
        document.getElementById("diagConfidence").textContent = data.confidence;

        // Severity badge
        const badge = document.getElementById("diagSeverityBadge");
        badge.textContent = data.clinical_info.severity;
        if (data.clinical_info.severity.toLowerCase().includes("high") || data.clinical_info.severity.toLowerCase().includes("malignant")) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-semibold bg-red-500/20 text-red-400 border border-red-500/30";
        } else if (data.clinical_info.severity.toLowerCase().includes("moderate")) {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30";
        } else {
            badge.className = "px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        }

        // Clinical guidance
        document.getElementById("diagGuidance").textContent = data.clinical_info.recommendations[0] || data.clinical_info.urgency;

        // Update probability chart
        if (probabilityChartInstance && data.probabilities) {
            const labels = Object.keys(data.probabilities);
            const vals = Object.values(data.probabilities);
            probabilityChartInstance.data.labels = labels;
            probabilityChartInstance.data.datasets[0].data = vals;
            probabilityChartInstance.update();
        }

        logToConsole(`[DIAGNOSIS] Pathology identified: ${data.disease_name} (${data.confidence}% confidence)`, "text-cyan-400");
    })
    .catch(err => {
        btn.disabled = false;
        logToConsole(`[ERROR] Inference failed: ${err.message}`, "text-rose-400");
    });
}

function setPipelineStep(stepNum) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`stepIndicator${i}`);
        if (el) {
            if (i <= stepNum) el.classList.add("active");
            else el.classList.remove("active");
        }
    }
}

function logToConsole(message, colorClass = "text-slate-300") {
    const consoleBox = document.getElementById("trainConsole");
    const p = document.createElement("p");
    p.className = `${colorClass} leading-snug`;
    const time = new Date().toLocaleTimeString();
    p.innerHTML = `<span class="text-slate-500">[${time}]</span> ${message}`;
    consoleBox.appendChild(p);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

function toggleHelpModal() {
    const modal = document.getElementById("helpModal");
    modal.classList.toggle("hidden");
}

