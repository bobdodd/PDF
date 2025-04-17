#!/usr/bin/env python3
"""
Example script that demonstrates how to batch process multiple images.
This script processes a directory of images and generates a summary report.
"""

import os
import sys
import argparse
import json
from tqdm import tqdm

# Add parent directory to path to import the library modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import FeatureExtractor
from src.ollama_integration import OllamaIntegration

def process_directory(directory, model_name="italic-detector", output_file=None):
    """Process all images in a directory and detect italic text."""
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create feature extractor
    extractor = FeatureExtractor()
    
    # Create Ollama integration
    ollama = OllamaIntegration(model_name=model_name)
    
    # Process each image
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract features
            features = extractor.extract_features_from_file(image_path)
            
            # Run prediction
            prediction = ollama.run_prediction(features.tolist())
            
            # Store result
            result = {
                'image': os.path.basename(image_path),
                'path': image_path,
                'is_italic': prediction.get('is_italic', False),
                'confidence': prediction.get('confidence', 0)
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Generate summary
    italic_count = sum(1 for r in results if r.get('is_italic'))
    regular_count = len(results) - italic_count
    
    print(f"\nAnalysis complete.")
    print(f"Found {italic_count} italic text images and {regular_count} regular text images.")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch process images for italic text detection.')
    parser.add_argument('directory', help='Directory containing images to process')
    parser.add_argument('--model-name', default='italic-detector', help='Name of the Ollama model to use')
    parser.add_argument('--output', help='Path to save results JSON file')
    
    args = parser.parse_args()
    process_directory(args.directory, args.model_name, args.output)