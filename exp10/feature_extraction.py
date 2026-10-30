"""
feature_extraction.py
Skin Disease Detection - Comprehensive Feature Extraction Module

Extracts 42 quantitative dermatological features across four domains:
1. Color Moments (RGB and HSV spaces: Mean, Standard Deviation, Skewness) -> 18 features
2. Texture Analysis (Gray-Level Co-occurrence Matrix / GLCM Haralick metrics) -> 6 features
3. Dermatological Shape / ABCD Rule (Asymmetry, Border irregularity, Compactness, Eccentricity, Solidity) -> 8 features
4. Micro-texture (Local Binary Patterns / LBP Histogram) -> 10 features
"""

import cv2
import numpy as np
from scipy import stats

class SkinFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            # Color RGB (9)
            "R_Mean", "G_Mean", "B_Mean",
            "R_Std", "G_Std", "B_Std",
            "R_Skew", "G_Skew", "B_Skew",
            # Color HSV (9)
            "H_Mean", "S_Mean", "V_Mean",
            "H_Std", "S_Std", "V_Std",
            "H_Skew", "S_Skew", "V_Skew",
            # GLCM Texture (6)
            "GLCM_Contrast", "GLCM_Dissimilarity", "GLCM_Homogeneity",
            "GLCM_Energy", "GLCM_Correlation", "GLCM_Entropy",
            # Shape & ABCD Metrics (8)
            "Asymmetry_Index", "Border_Circularity", "Solidity",
            "Aspect_Ratio", "Extent", "Eccentricity", "Hu_Moment_1", "Hu_Moment_2",
            # LBP Micro-texture (10)
            "LBP_Bin_0", "LBP_Bin_1", "LBP_Bin_2", "LBP_Bin_3", "LBP_Bin_4",
            "LBP_Bin_5", "LBP_Bin_6", "LBP_Bin_7", "LBP_Bin_8", "LBP_NonUniform"
        ]

    def extract_color_moments(self, image_bgr, mask=None):
        """
        Extracts 1st (mean), 2nd (std), and 3rd (skewness) order color moments
        from RGB and HSV color spaces within the segmented lesion mask.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        if mask is not None and np.sum(mask) > 0:
            roi_pixels_rgb = image_rgb[mask > 0]
            roi_pixels_hsv = image_hsv[mask > 0]
        else:
            roi_pixels_rgb = image_rgb.reshape(-1, 3)
            roi_pixels_hsv = image_hsv.reshape(-1, 3)

        features = []
        
        # RGB moments
        for channel in range(3):
            c_data = roi_pixels_rgb[:, channel].astype(float)
            mean = float(np.mean(c_data))
            std = float(np.std(c_data)) + 1e-6
            skew = float(stats.skew(c_data)) if len(c_data) > 2 else 0.0
            if np.isnan(skew):
                skew = 0.0
            features.extend([mean, std, skew])

        # HSV moments
        for channel in range(3):
            c_data = roi_pixels_hsv[:, channel].astype(float)
            mean = float(np.mean(c_data))
            std = float(np.std(c_data)) + 1e-6
            skew = float(stats.skew(c_data)) if len(c_data) > 2 else 0.0
            if np.isnan(skew):
                skew = 0.0
            features.extend([mean, std, skew])

        return features

    def compute_glcm(self, gray_img, levels=16):
        """
        Computes 2D Gray-Level Co-occurrence Matrix (GLCM) quantized to 16 levels
        averaged over 4 directions (0, 45, 90, 135 degrees) with distance d=1.
        """
        # Quantize image from 256 to `levels` bins
        quantized = (gray_img / (256.0 / levels)).astype(np.uint8)
        quantized = np.clip(quantized, 0, levels - 1)
        h, w = quantized.shape
        
        glcm = np.zeros((levels, levels), dtype=float)
        
        # Directions: (dx, dy)
        offsets = [(0, 1), (1, 1), (1, 0), (1, -1)]
        
        for dy, dx in offsets:
            for y in range(max(0, -dy), min(h, h - dy)):
                for x in range(max(0, -dx), min(w, w - dx)):
                    row = quantized[y, x]
                    col = quantized[y + dy, x + dx]
                    glcm[row, col] += 1.0
                    glcm[col, row] += 1.0  # symmetric
        
        # Normalize to joint probability distribution
        total = np.sum(glcm)
        if total > 0:
            glcm /= total
        else:
            glcm.fill(1.0 / (levels * levels))
            
        return glcm

    def extract_glcm_features(self, gray_img, mask=None):
        """
        Extracts 6 standard Haralick texture features from GLCM:
        Contrast, Dissimilarity, Homogeneity, Energy (ASM), Correlation, Entropy.
        """
        if mask is not None and np.sum(mask) > 0:
            # Crop to bounding box of mask for focused texture analysis
            y_indices, x_indices = np.where(mask > 0)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            roi_gray = gray_img[y_min:y_max+1, x_min:x_max+1]
        else:
            roi_gray = gray_img
            
        if roi_gray.shape[0] < 4 or roi_gray.shape[1] < 4:
            roi_gray = gray_img

        levels = 16
        P = self.compute_glcm(roi_gray, levels=levels)
        
        i, j = np.indices((levels, levels))
        
        # Contrast
        contrast = float(np.sum(P * ((i - j) ** 2)))
        
        # Dissimilarity
        dissimilarity = float(np.sum(P * np.abs(i - j)))
        
        # Homogeneity
        homogeneity = float(np.sum(P / (1.0 + (i - j) ** 2)))
        
        # Energy / Angular Second Moment (ASM)
        energy = float(np.sum(P ** 2))
        
        # Correlation
        mu_i = np.sum(i * P)
        mu_j = np.sum(j * P)
        var_i = np.sum(((i - mu_i) ** 2) * P)
        var_j = np.sum(((j - mu_j) ** 2) * P)
        std_i = np.sqrt(var_i) if var_i > 0 else 1.0
        std_j = np.sqrt(var_j) if var_j > 0 else 1.0
        correlation = float(np.sum((i - mu_i) * (j - mu_j) * P) / (std_i * std_j))
        
        # Entropy
        non_zero = P[P > 0]
        entropy = float(-np.sum(non_zero * np.log2(non_zero)))
        
        return [contrast, dissimilarity, homogeneity, energy, correlation, entropy]

    def extract_shape_features(self, contour, mask):
        """
        Extracts 8 dermatological shape & ABCD criteria features:
        Asymmetry Index, Border Circularity, Solidity, Aspect Ratio,
        Extent, Eccentricity, and Hu Moments (h1, h2).
        """
        if contour is None or len(contour) < 5 or cv2.contourArea(contour) < 10:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, closed=True))
        
        # 1. Border Circularity / Compactness: 4*pi*Area / (Perimeter^2)
        circularity = float(4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        circularity = min(circularity, 1.0)
        
        # 2. Solidity: Area / ConvexHullArea
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        
        # 3. Bounding box & Aspect Ratio, Extent
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        extent = float(area) / float(w * h) if (w * h) > 0 else 0.0
        
        # 4. Asymmetry Index: Compare horizontal & vertical halves of the mask ROI
        roi_mask = mask[y:y+h, x:x+w]
        h_half, w_half = roi_mask.shape[0] // 2, roi_mask.shape[1] // 2
        
        top_half = roi_mask[:h_half, :]
        bot_half = np.flipud(roi_mask[roi_mask.shape[0]-h_half:, :])
        diff_v = np.sum(np.bitwise_xor(top_half, bot_half))
        
        left_half = roi_mask[:, :w_half]
        right_half = np.fliplr(roi_mask[:, roi_mask.shape[1]-w_half:])
        diff_h = np.sum(np.bitwise_xor(left_half, right_half))
        
        asymmetry_index = float((diff_v + diff_h) / (2.0 * area)) if area > 0 else 0.0
        
        # 5. Eccentricity from moments or fitted ellipse
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, orientation) = ellipse
                major_axis = max(axes)
                minor_axis = min(axes)
                if major_axis > 0:
                    eccentricity = float(np.sqrt(1.0 - (minor_axis / major_axis) ** 2))
            except Exception:
                eccentricity = 0.0

        # 6. Hu Moments
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        # Log scale Hu moments for numerical stability
        hu_log1 = float(-np.sign(hu[0]) * np.log10(np.abs(hu[0]) + 1e-12))
        hu_log2 = float(-np.sign(hu[1]) * np.log10(np.abs(hu[1]) + 1e-12))

        return [
            asymmetry_index, circularity, solidity,
            aspect_ratio, extent, eccentricity, hu_log1, hu_log2
        ]

    def extract_lbp_features(self, gray_img, mask=None):
        """
        Extracts 10-bin Local Binary Pattern (LBP) histogram representing surface micro-texture.
        Uniform patterns (P=8, R=1).
        """
        h, w = gray_img.shape
        # Compute 8-neighbor LBP
        lbp_img = np.zeros((h - 2, w - 2), dtype=np.uint8)
        
        center = gray_img[1:-1, 1:-1]
        
        # Neighbors (8 directions clockwise)
        neighbors = [
            gray_img[:-2, :-2],  # top-left
            gray_img[:-2, 1:-1], # top
            gray_img[:-2, 2:],   # top-right
            gray_img[1:-1, 2:],  # right
            gray_img[2:, 2:],    # bot-right
            gray_img[2:, 1:-1],  # bot
            gray_img[2:, :-2],   # bot-left
            gray_img[1:-1, :-2]  # left
        ]
        
        for i, n in enumerate(neighbors):
            lbp_img += ((n >= center).astype(np.uint8) << i)
            
        # Map to uniform LBP patterns (patterns with <= 2 bit transitions 0->1 or 1->0)
        # Fast lookup table
        lut = np.zeros(256, dtype=int)
        uniform_idx = 0
        for code in range(256):
            binary_str = f"{code:08b}"
            transitions = sum(binary_str[b] != binary_str[(b + 1) % 8] for b in range(8))
            if transitions <= 2:
                lut[code] = uniform_idx
                uniform_idx += 1
            else:
                lut[code] = 9  # Non-uniform bin
                
        # Remap
        uniform_lbp = lut[lbp_img]
        
        if mask is not None and np.sum(mask) > 0:
            roi_mask = mask[1:-1, 1:-1]
            valid_pixels = uniform_lbp[roi_mask > 0]
        else:
            valid_pixels = uniform_lbp.flatten()

        hist, _ = np.histogram(valid_pixels, bins=10, range=(0, 10), density=True)
        return [float(x) for x in hist]

    def extract_all_features(self, image_bgr, mask=None, contour=None):
        """
        Combines all four categories into a single feature vector of 42 numerical values.
        Returns:
            feature_vector: list of 42 floats
            feature_dict: dict mapping feature name -> float value
            feature_breakdown: categorized dictionary
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        color_feats = self.extract_color_moments(image_bgr, mask)
        glcm_feats = self.extract_glcm_features(gray, mask)
        shape_feats = self.extract_shape_features(contour, mask if mask is not None else np.ones_like(gray))
        lbp_feats = self.extract_lbp_features(gray, mask)
        
        full_vector = color_feats + glcm_feats + shape_feats + lbp_feats
        
        # Clean any accidental NaNs or Infs
        full_vector = [0.0 if (np.isnan(v) or np.isinf(v)) else round(float(v), 4) for v in full_vector]
        
        feature_dict = dict(zip(self.feature_names, full_vector))
        
        breakdown = {
            "Color Moments (RGB & HSV)": dict(zip(self.feature_names[:18], full_vector[:18])),
            "GLCM Texture Analysis": dict(zip(self.feature_names[18:24], full_vector[18:24])),
            "Shape & ABCD Metrics": dict(zip(self.feature_names[24:32], full_vector[24:32])),
            "LBP Micro-texture Histogram": dict(zip(self.feature_names[32:42], full_vector[32:42]))
        }
        
        return full_vector, feature_dict, breakdown

