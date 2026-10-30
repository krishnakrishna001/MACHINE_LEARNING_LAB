"""
test_system.py
Automated validation of all 5 pipeline components and Flask web routes.
"""

import warnings
warnings.filterwarnings('ignore')

import unittest
import os
import cv2
import numpy as np
from app import app
from preprocessor import SkinPreprocessor
from feature_extraction import SkinFeatureExtractor
from train_model import SkinModelTrainer

class TestSkinDiseaseDetection(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.preprocessor = SkinPreprocessor()
        self.extractor = SkinFeatureExtractor()

    def test_01_preprocessing_pipeline(self):
        print("\n[TEST 1] Testing Image Preprocessing Pipeline...")
        sample_path = "static/samples/melanoma.jpg"
        self.assertTrue(os.path.exists(sample_path), "Sample image not found!")
        
        img = cv2.imread(sample_path)
        self.assertIsNotNone(img, "Failed to read sample image")

        res = self.preprocessor.process_full_pipeline(img)
        self.assertIn("original", res)
        self.assertIn("hair_removed", res)
        self.assertIn("denoised", res)
        self.assertIn("enhanced_clahe", res)
        self.assertIn("binary_mask", res)
        self.assertIn("contour_overlay", res)
        self.assertEqual(res["original"].shape, (256, 256, 3))
        print("  --> Preprocessing PASSED (All 6 intermediate stages generated).")

    def test_02_feature_extraction(self):
        print("[TEST 2] Testing 42-Feature Extraction...")
        img = cv2.imread("static/samples/melanoma.jpg")
        processed = self.preprocessor.process_full_pipeline(img)

        feats, feat_dict, breakdown = self.extractor.extract_all_features(
            processed["enhanced_clahe"],
            processed["mask_single_channel"],
            processed["contour"]
        )
        self.assertEqual(len(feats), 42, f"Expected 42 features, got {len(feats)}")
        self.assertEqual(len(feat_dict), 42)
        self.assertIn("Color Moments (RGB & HSV)", breakdown)
        self.assertIn("GLCM Texture Analysis", breakdown)
        self.assertIn("Shape & ABCD Metrics", breakdown)
        self.assertIn("LBP Micro-texture Histogram", breakdown)
        print("  --> Feature Extraction PASSED (Extracted 42 features across 4 clinical domains).")

    def test_03_flask_routes(self):
        print("[TEST 3] Testing Flask Web Application Endpoints...")
        # GET /
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"DermAI", resp.data)
        print("  --> GET / (One-Frame Dashboard) PASSED.")

        # POST /api/process-image
        resp = self.client.post('/api/process-image', json={"sample_file": "melanoma.jpg"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertIn("images", data)
        self.assertIn("features_summary", data)
        print("  --> POST /api/process-image PASSED.")

        # POST /api/predict
        resp = self.client.post('/api/predict', json={})
        self.assertEqual(resp.status_code, 200)
        pred_data = resp.get_json()
        self.assertTrue(pred_data["success"])
        self.assertIn("disease_name", pred_data)
        print(f"  --> POST /api/predict PASSED (Predicted: {pred_data['disease_name']}, Confidence: {pred_data['confidence']}%).")

        # GET /api/benchmark
        resp = self.client.get('/api/benchmark')
        self.assertEqual(resp.status_code, 200)
        bench_data = resp.get_json()
        self.assertTrue(bench_data["success"])
        print(f"  --> GET /api/benchmark PASSED ({len(bench_data['benchmark'])} models compared).")

if __name__ == '__main__':
    unittest.main()
