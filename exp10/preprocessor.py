"""
preprocessor.py
Skin Disease Detection - Image Preprocessing Module

Provides comprehensive dermatological image preprocessing:
1. Standard Resizing
2. Dull Razor Algorithm (Morphological Hair Removal + Inpainting)
3. Noise Reduction (Gaussian & Median Filtering)
4. Contrast Enhancement (CLAHE - Contrast Limited Adaptive Histogram Equalization)
5. Color Space Conversions (RGB -> HSV, LAB, Grayscale)
6. Lesion Segmentation (Otsu Thresholding, Morphological Cleanup, Contour Detection)
"""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class SkinPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def resize_image(self, img):
        """Resize image to target size while maintaining aspect ratio or padding."""
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

    def dull_razor_hair_removal(self, img):
        """
        Dull Razor algorithm:
        1. Convert to grayscale.
        2. Apply Black-Hat morphological filter with rectangular/cross kernels
           to detect dark thin hair artifacts.
        3. Threshold the black-hat result to produce hair mask.
        4. Inpaint using Fast Marching (Telea) algorithm.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Black hat filter with kernels in multiple orientations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to create hair mask
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint hair regions using neighborhood skin pixels
        hair_removed = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return hair_removed, hair_mask

    def denoise_image(self, img):
        """Apply Gaussian & bilateral or median filtering to eliminate camera noise."""
        # Gaussian smoothing preserves edges better with 5x5
        denoised = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
        return denoised

    def apply_clahe(self, img):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the L (Luminance) channel in LAB color space to enhance lesion details.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def segment_lesion(self, img_preprocessed):
        """
        Segment the skin lesion using Otsu's thresholding, morphological closing/opening,
        and contour detection.
        Returns:
            binary_mask: 0 or 255 binary image of the lesion
            segmented_lesion: Lesion cropped on black background
            contour_overlay: Original image with green contour bounding the lesion
            contour: The primary lesion contour coordinates
        """
        # Blue channel or Grayscale usually has highest lesion contrast
        gray = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2GRAY)
        
        # Invert if skin is bright and lesion is dark
        # Otsu thresholding automatically computes optimal bimodal threshold
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours and choose the most prominent central contour
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        primary_contour = None
        
        if contours:
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Pick largest contour that is reasonably central and large enough
            primary_contour = contours[0]
            cv2.drawContours(mask, [primary_contour], -1, 255, thickness=cv2.FILLED)
        else:
            # Fallback circular central mask
            h, w = gray.shape
            cv2.circle(mask, (w//2, h//2), min(h, w)//3, 255, -1)
            primary_contour = np.array([[[w//2, h//2]]])

        # Masked image (lesion only)
        masked_lesion = cv2.bitwise_and(img_preprocessed, img_preprocessed, mask=mask)
        
        # Contour overlay on color image
        contour_overlay = img_preprocessed.copy()
        if primary_contour is not None and len(primary_contour) > 2:
            cv2.drawContours(contour_overlay, [primary_contour], -1, (0, 255, 0), 2)
            # Add bounding box or minimum enclosing circle
            x, y, w_box, h_box = cv2.boundingRect(primary_contour)
            cv2.rectangle(contour_overlay, (x, y), (x + w_box, y + h_box), (0, 165, 255), 1)

        return mask, masked_lesion, contour_overlay, primary_contour

    def process_full_pipeline(self, image_bgr):
        """
        Executes the complete preprocessing pipeline on an input image:
        Returns a dictionary of all intermediate images for visualization.
        """
        # Step 1: Resize
        resized = self.resize_image(image_bgr)
        
        # Step 2: Dull Razor Hair Removal
        hair_removed, hair_mask = self.dull_razor_hair_removal(resized)
        
        # Step 3: Denoising
        denoised = self.denoise_image(hair_removed)
        
        # Step 4: Contrast Enhancement (CLAHE)
        enhanced = self.apply_clahe(denoised)
        
        # Step 5: Color spaces
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Step 6: Lesion Segmentation
        mask, masked_lesion, contour_overlay, contour = self.segment_lesion(enhanced)
        
        return {
            "original": resized,
            "hair_mask": cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR),
            "hair_removed": hair_removed,
            "denoised": denoised,
            "enhanced_clahe": enhanced,
            "hsv_view": hsv,
            "lab_view": lab,
            "grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            "binary_mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "masked_lesion": masked_lesion,
            "contour_overlay": contour_overlay,
            "contour": contour,
            "mask_single_channel": mask
        }

    @staticmethod
    def encode_image_base64(img_bgr):
        """Encodes OpenCV BGR image into base64 JPEG string for web display."""
        if img_bgr is None:
            return ""
        _, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_str}"

