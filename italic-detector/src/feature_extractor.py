import numpy as np
import cv2
from typing import List, Tuple

class FeatureExtractor:
    """Extracts features from text images for italic detection."""
    
    def __init__(self, 
                 resize_shape: Tuple[int, int] = (100, 32),
                 angle_bins: int = 18):
        """Initialize the feature extractor with parameters."""
        self.resize_shape = resize_shape
        self.angle_bins = angle_bins
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image for italic detection."""
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast and brightness normalization
        # First, apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Further normalize to ensure consistent brightness range
        # Scale to 0-255 range to standardize brightness
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:  # Avoid division by zero
            image = np.uint8(255 * ((image - min_val) / (max_val - min_val)))
        
        # Trim whitespace to get a tight bounding box around the text
        # Apply Otsu thresholding to separate text from background
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find non-zero pixels (text)
        coords = cv2.findNonZero(binary)
        
        if coords is not None and len(coords) > 0:
            # Get the bounding box of non-zero pixels
            x, y, w, h = cv2.boundingRect(coords)
            
            # Crop to the text region with a small margin (2 pixels)
            margin = 2
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(image.shape[1], x + w + margin)
            y_max = min(image.shape[0], y + h + margin)
            
            # Crop the image to the text region
            image = image[y_min:y_max, x_min:x_max]
        
        # Resize to standard dimensions
        img_resized = cv2.resize(image, self.resize_shape)
        
        # Apply aggressive binarization to standardize line thickness
        # Use adaptive thresholding which handles local variations better
        binary = cv2.adaptiveThreshold(
            img_resized, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size for local threshold calculation
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations to standardize line thickness
        # Create a small kernel for morphological operations
        kernel = np.ones((2, 2), np.uint8)
        
        # Erode to reduce noise and thin lines slightly
        binary = cv2.erode(binary, kernel, iterations=1)
        
        # Then dilate to standardize line thickness
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find edges
        edges = cv2.Canny(binary, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 
                             minLineLength=5, maxLineGap=3)
        
        # Calculate angles of detected lines
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = np.arctan((y2 - y1) / (x2 - x1))
                    angles.append(angle)
        
        # Handle cases with no detected lines
        if not angles:
            # If no lines are detected, use a fallback feature set
            hist = np.zeros(self.angle_bins)
            pixel_density = np.sum(binary < 128) / (binary.shape[0] * binary.shape[1])
            diag_ratio = 0.5  # Neutral value
            diag_length_ratio = 0.5  # Neutral value for length ratio
            diag_difference = 0  # No difference for fallback
            total_lines = 1  # Prevent division by zero
            pos_diag_ratio = 0.5  # Neutral value
            neg_diag_ratio = 0.5  # Neutral value
        else:
            # Create histogram of angles
            hist, _ = np.histogram(angles, bins=self.angle_bins, 
                                range=(-np.pi/2, np.pi/2))
            
            # Normalize histogram
            hist = hist / (np.sum(hist) + 1e-10)  # Avoid division by zero
            
            # Calculate additional features
            pixel_density = np.sum(binary < 128) / (binary.shape[0] * binary.shape[1])
            
            # Count diagonal (positive/negative) vs. vertical edges
            pos_diag_count = 0  # Bottom-left to top-right (positive slope, typical italic)
            neg_diag_count = 0  # Top-left to bottom-right (negative slope, like in 'A')
            vert_count = 0
            
            # Calculate line lengths for scale-invariant features
            pos_diag_length = 0
            neg_diag_length = 0
            vert_length = 0
            total_length = 0
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate line length
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    total_length += length
                    
                    if x2 - x1 != 0:  # Avoid division by zero
                        angle = np.arctan((y2 - y1) / (x2 - x1))
                        
                        # Positive diagonals (angle > 0): bottom-left to top-right, typical italic slant
                        if angle > 0.25 and angle < 1.30:  # ~14 to ~74 degrees - more conservative
                            pos_diag_count += 1
                            pos_diag_length += length
                        # Negative diagonals (angle < 0): top-left to bottom-right, like in 'A'
                        elif angle < -0.25 and angle > -1.30:  # ~-14 to ~-74 degrees
                            neg_diag_count += 1
                            neg_diag_length += length
                        else:
                            vert_count += 1
                            vert_length += length
            
            # Calculate metrics that help distinguish true italic from 'A' characters
            # Emphasis on positive diagonals (italic slant) while accounting for negative diagonals 
            diag_difference = pos_diag_count - neg_diag_count
            total_lines = pos_diag_count + neg_diag_count + vert_count + 1e-10
            
            # Scale-invariant features based on line length ratios
            total_length = max(total_length, 1e-10)  # Prevent division by zero
            pos_diag_ratio = pos_diag_length / total_length
            neg_diag_ratio = neg_diag_length / total_length
            
            # Adjusted ratio that reduces false positives with 'A' characters
            # Using both count-based and length-based features for robustness
            diag_ratio = (pos_diag_count - 0.75 * neg_diag_count) / total_lines
            diag_length_ratio = (pos_diag_length - 0.75 * neg_diag_length) / total_length
        
        # Calculate additional features for better detection
        
        # Average slant angle - useful for italic detection
        if angles:
            avg_slant = np.mean([abs(angle) for angle in angles])
        else:
            avg_slant = 0
            
        # Calculate image moments to detect text slant
        moments = cv2.moments(binary)
        if moments['m00'] != 0:
            # Centroid coordinates
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            # Second order moments for orientation
            orientation_numerator = 2 * moments['mu11']
            orientation_denominator = moments['mu20'] - moments['mu02']
            
            if orientation_denominator != 0:
                theta = 0.5 * np.arctan2(orientation_numerator, orientation_denominator)
                orientation = theta
            else:
                orientation = 0
        else:
            orientation = 0
        
        # Combine all features
        additional_features = np.array([
            pixel_density,
            diag_ratio,
            diag_length_ratio,  # Scale-invariant length-based ratio
            avg_slant,
            abs(orientation),
            np.sin(2 * abs(orientation)),  # Good indicator for slant
            diag_difference / (total_lines if 'total_lines' in locals() else 1e-10),  # Difference between positive and negative diagonals
            pos_diag_ratio if 'pos_diag_ratio' in locals() else 0.5,  # Scale-invariant positive diagonal ratio
            neg_diag_ratio if 'neg_diag_ratio' in locals() else 0.5   # Scale-invariant negative diagonal ratio
        ])
        
        return np.concatenate([hist, additional_features])
    
    def extract_features_from_file(self, image_path: str) -> np.ndarray:
        """Extract features from an image file."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.extract_features(image)