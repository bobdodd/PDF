import os
import sys
import unittest
import numpy as np
import cv2

# Add parent directory to path to import the library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    """Tests for the FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()
        
        # Create test images
        # 1. Blank image
        self.blank_image = np.ones((32, 100), dtype=np.uint8) * 255
        
        # 2. Vertical lines image (regular text-like)
        self.vertical_image = np.ones((32, 100), dtype=np.uint8) * 255
        for i in range(10, 90, 10):
            cv2.line(self.vertical_image, (i, 5), (i, 25), 0, 1)
        
        # 3. Slanted lines image (italic text-like)
        self.italic_image = np.ones((32, 100), dtype=np.uint8) * 255
        for i in range(10, 90, 10):
            cv2.line(self.italic_image, (i, 5), (i+5, 25), 0, 1)
            
    def test_extract_features_shape(self):
        """Test that feature extraction produces correct shape."""
        features = self.extractor.extract_features(self.blank_image)
        
        # Default angle_bins is 18, plus 5 additional features
        expected_length = 18 + 5
        self.assertEqual(len(features), expected_length)
    
    def test_extract_features_blank(self):
        """Test feature extraction on blank image."""
        features = self.extractor.extract_features(self.blank_image)
        
        # Blank image should have very low/no diagonal ratio
        diag_ratio = features[19]  # diag_ratio is the 20th feature (index 19)
        self.assertLessEqual(diag_ratio, 0.5)
    
    def test_extract_features_vertical(self):
        """Test feature extraction on vertical lines (regular text-like)."""
        features = self.extractor.extract_features(self.vertical_image)
        
        # Vertical lines should have lower diagonal ratio
        diag_ratio = features[19]
        self.assertLessEqual(diag_ratio, 0.5)
    
    def test_extract_features_italic(self):
        """Test feature extraction on slanted lines (italic text-like)."""
        features = self.extractor.extract_features(self.italic_image)
        
        # Italic-like image should have higher diagonal ratio and slant angle
        diag_ratio = features[19]
        avg_slant = features[20]
        
        # These should show some diagonal features
        self.assertGreaterEqual(diag_ratio + avg_slant, 0.2)

if __name__ == '__main__':
    unittest.main()