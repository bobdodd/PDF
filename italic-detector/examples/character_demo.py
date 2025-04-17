#!/usr/bin/env python3
"""
Character-Level Italic Detection Demo Script

This script demonstrates how to use the character-level italic detection system
to analyze text images at the character level.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.character_segmentation import CharacterSegmentation
from src.character_feature_extractor import CharacterFeatureExtractor
from src.character_model import CharacterModel

def main():
    """Main function for character-level detection demo."""
    parser = argparse.ArgumentParser(description='Character-Level Italic Detection Demo')
    parser.add_argument('image_path', type=str, help='Path to text image to analyze')
    parser.add_argument('--char-threshold', type=float, default=0.60,
                      help='Confidence threshold for character-level detection')
    parser.add_argument('--output-path', type=str, default=None,
                      help='Path to save visualization (default: input_path_detection.png)')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Set output path
    if args.output_path is None:
        args.output_path = os.path.splitext(args.image_path)[0] + "_detection.png"
    
    # Initialize components
    segmenter = CharacterSegmentation()
    feature_extractor = CharacterFeatureExtractor()
    model = CharacterModel()
    
    # Try to load models
    try:
        model.load_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Using feature visualization only (no classification)")
        model = None
    
    # Read image
    print(f"Reading image: {args.image_path}")
    image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read image {args.image_path}")
        return
    
    # Segment characters
    print("Segmenting characters...")
    characters = segmenter.segment_characters(image)
    if not characters:
        print("No characters detected in the image.")
        return
    
    print(f"Detected {len(characters)} characters")
    
    # Create visualization
    vis_image = segmenter.visualize_segmentation(image, characters)
    
    # Extract features and classify if model is available
    if model and model.character_model:
        print("Extracting features and classifying characters...")
        char_features = []
        for char in characters:
            features = feature_extractor.extract_character_features(char['image'])
            char_features.append(features)
        
        # Make predictions
        X_char = np.vstack(char_features)
        char_probas = model.character_model.predict_proba(X_char)
        char_predictions = []
        
        for i, proba in enumerate(char_probas):
            is_italic = proba[1] >= args.char_threshold
            char_predictions.append({
                'position': i,
                'is_italic': bool(is_italic),
                'confidence': float(proba[1])
            })
        
        # Count italic characters
        italic_count = sum(1 for pred in char_predictions if pred['is_italic'])
        print(f"Result: {italic_count} of {len(characters)} characters are italic")
        print(f"Character-level italic percentage: {italic_count / len(characters) * 100:.1f}%")
        
        # Determine overall word status
        word_is_italic = italic_count / len(characters) >= 0.5
        print(f"Overall word classification: {'ITALIC' if word_is_italic else 'REGULAR'}")
        
        # Add predictions to visualization
        for i, char in enumerate(characters):
            x, y, w, h = char['bbox']
            if char_predictions[i]['is_italic']:
                color = (0, 0, 255)  # Red for italic
                label = "I"
            else:
                color = (0, 255, 0)  # Green for regular
                label = "R"
            
            # Draw label
            cv2.putText(vis_image, label, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw confidence
            conf_str = f"{char_predictions[i]['confidence']:.2f}"
            cv2.putText(vis_image, conf_str, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Also add overall result to image
        cv2.putText(vis_image, f"Overall: {'ITALIC' if word_is_italic else 'REGULAR'}", 
                   (10, image.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if word_is_italic else (0, 255, 0), 1)
    
    # Save visualization
    print(f"Saving visualization to {args.output_path}")
    cv2.imwrite(args.output_path, vis_image)
    
    # Display features for first character
    if len(characters) > 0:
        print("\nShowing features for first character:")
        char = characters[0]
        features = feature_extractor.extract_character_features(char['image'])
        
        # Print some key features
        feature_names = [
            "Aspect ratio", "Pixel density", "Diagonal ratio", 
            "Diagonal length ratio", "Average slant", "Orientation",
            "Sine orientation", "X skewness"
        ]
        
        for name, value in zip(feature_names, features[-len(feature_names):]):
            print(f"  {name}: {value:.4f}")
        
        # Show character image
        plt.figure(figsize=(8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(char['image'], cmap='gray')
        plt.title("Character Image")
        plt.axis('off')
        
        # Show angle histogram
        plt.subplot(1, 2, 2)
        angle_bins = features[:18]  # First 18 features are angle histogram
        plt.bar(range(len(angle_bins)), angle_bins)
        plt.title("Angle Distribution")
        plt.xlabel("Angle Bin")
        plt.ylabel("Frequency")
        
        # Save figure
        plt.tight_layout()
        feature_viz_path = os.path.splitext(args.output_path)[0] + "_features.png"
        plt.savefig(feature_viz_path)
        print(f"Feature visualization saved to {feature_viz_path}")
        
        try:
            plt.show()
        except Exception:
            print("Could not display plot interactively.")
    
    print("Demo complete!")

if __name__ == "__main__":
    main()