#!/usr/bin/env python3
"""
Initialization script for the Italic Text Detector project.
This script sets up the required directories and basic test data.
"""

import os
import cv2
import numpy as np
import argparse

def create_directories():
    """Create the necessary directories for the project."""
    directories = [
        "data/italic",
        "data/regular",
        "data/processed",
        "models/saved",
        "examples",
        "tests"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_test_data():
    """Create sample test data for development purposes."""
    print("Creating sample test data...")
    
    # Create sample images for testing
    
    # 1. Regular text samples
    for i in range(10):
        # Create a white image
        image = np.ones((32, 100), dtype=np.uint8) * 255
        
        # Add vertical lines (simulating regular text)
        for j in range(10, 90, 10):
            cv2.line(image, (j, 5), (j, 25), 0, 1)
            
        # Add random noise
        noise = np.random.randint(0, 30, size=image.shape, dtype=np.uint8)
        image = cv2.subtract(image, noise)
        
        # Save the image
        cv2.imwrite(f"data/regular/sample_regular_{i}.png", image)
    
    # 2. Italic text samples
    for i in range(10):
        # Create a white image
        image = np.ones((32, 100), dtype=np.uint8) * 255
        
        # Add slanted lines (simulating italic text)
        for j in range(10, 90, 10):
            cv2.line(image, (j, 5), (j+5, 25), 0, 1)
            
        # Add random noise
        noise = np.random.randint(0, 30, size=image.shape, dtype=np.uint8)
        image = cv2.subtract(image, noise)
        
        # Save the image
        cv2.imwrite(f"data/italic/sample_italic_{i}.png", image)
        
    print(f"Created 10 sample regular images in data/regular/")
    print(f"Created 10 sample italic images in data/italic/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize the Italic Text Detector project.')
    parser.add_argument('--sample-data', action='store_true', 
                       help='Create sample data for testing')
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Create sample data if requested
    if args.sample_data:
        create_test_data()
        
    print("\nInitialization complete. You can now:")
    print("1. Run 'python init.py --sample-data' to generate test data")
    print("2. Install the package with 'pip install -e .'")
    print("3. Train the model with 'italic-detector prepare-data' followed by 'italic-detector train-model'")
    print("4. Deploy the model to Ollama with 'italic-detector deploy-to-ollama'")
    print("5. Test the model with 'italic-detector detect-italic --image-path data/italic/sample_italic_0.png'")