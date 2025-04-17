#!/usr/bin/env python3
"""
Example script that demonstrates how to use the italic detector programmatically.
This script analyzes a single image and determines if it contains italic text.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import the library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.ollama_integration import OllamaIntegration

def analyze_image(image_path, model_name="italic-detector", show_visualization=False):
    """Analyze an image to determine if it contains italic text."""
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Extract features from image
    try:
        features = extractor.extract_features_from_file(image_path)
        
        # Create Ollama integration
        ollama = OllamaIntegration(model_name=model_name)
        
        # Run prediction
        print(f"Analyzing text in {image_path}...")
        result = ollama.run_prediction(features.tolist())
        
        # Display result
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        else:
            is_italic = result.get('is_italic', False)
            confidence = result.get('confidence', 0)
            
            print(f"Result: {'ITALIC' if is_italic else 'REGULAR'} text")
            print(f"Confidence: {confidence:.2f}")
            
            # Optional visualization
            if show_visualization:
                import cv2
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title('Original Image')
                
                # Get angle histogram
                angle_hist = features[:18]  # First 18 features are angle histogram
                
                plt.subplot(1, 2, 2)
                plt.bar(range(len(angle_hist)), angle_hist)
                plt.title('Angle Histogram')
                plt.xlabel('Angle Bins')
                plt.ylabel('Frequency')
                
                # Add text with prediction and confidence
                plt.figtext(0.5, 0.01, 
                           f"Prediction: {'ITALIC' if is_italic else 'REGULAR'} (Confidence: {confidence:.2f})",
                           ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
                
                plt.tight_layout()
                plt.show()
            
            return is_italic
            
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze image for italic text.')
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('--model-name', default='italic-detector', help='Name of the Ollama model to use')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of analysis')
    
    args = parser.parse_args()
    analyze_image(args.image_path, args.model_name, args.visualize)