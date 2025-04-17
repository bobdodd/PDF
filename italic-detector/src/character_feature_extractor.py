import numpy as np
import cv2
from typing import List, Dict, Tuple, Union
from .feature_extractor import FeatureExtractor

class CharacterFeatureExtractor:
    """Extracts features from individual character images for italic detection."""
    
    def __init__(self, 
                 resize_shape: Tuple[int, int] = (32, 32),
                 angle_bins: int = 18):
        """Initialize the character feature extractor with parameters.
        
        Args:
            resize_shape: Size to resize each character image to
            angle_bins: Number of bins for angle histogram
        """
        self.resize_shape = resize_shape
        self.angle_bins = angle_bins
        self.word_feature_extractor = FeatureExtractor()
        
    def extract_character_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a character image for italic detection.
        
        Args:
            image: Grayscale image of a single character
            
        Returns:
            Feature vector for the character
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
        # Scale to 0-255 range
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:  # Avoid division by zero
            image = np.uint8(255 * ((image - min_val) / (max_val - min_val)))
        
        # Resize to standard dimensions
        img_resized = cv2.resize(image, self.resize_shape)
        
        # Binarize using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img_resized, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        # Find edges
        edges = cv2.Canny(binary, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, 
                               minLineLength=5, maxLineGap=3)
        
        # Calculate character-specific features
        
        # 1. Character height-to-width ratio
        height, width = binary.shape
        aspect_ratio = height / max(width, 1)  # Avoid division by zero
        
        # 2. Calculate angles of detected lines
        angles = []
        pos_diag_count = 0
        neg_diag_count = 0
        pos_diag_length = 0
        neg_diag_length = 0
        vert_count = 0
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
                    angles.append(angle)
                    
                    # Positive diagonals (bottom-left to top-right)
                    if angle > 0.25 and angle < 1.30:
                        pos_diag_count += 1
                        pos_diag_length += length
                    # Negative diagonals (top-left to bottom-right)
                    elif angle < -0.25 and angle > -1.30:
                        neg_diag_count += 1
                        neg_diag_length += length
                    else:
                        vert_count += 1
                        vert_length += length
        
        # 3. Create angle histogram
        hist = np.zeros(self.angle_bins)
        if angles:
            hist, _ = np.histogram(angles, bins=self.angle_bins, 
                                  range=(-np.pi/2, np.pi/2))
            hist = hist / max(np.sum(hist), 1e-10)  # Normalize
        
        # 4. Calculate average slant angle
        avg_slant = np.mean([abs(angle) for angle in angles]) if angles else 0
        
        # 5. Calculate diagonal ratios
        total_lines = pos_diag_count + neg_diag_count + vert_count + 1e-10
        diag_ratio = (pos_diag_count - 0.75 * neg_diag_count) / total_lines
        
        total_length = max(total_length, 1e-10)
        diag_length_ratio = (pos_diag_length - 0.75 * neg_diag_length) / total_length
        
        # 6. Calculate pixel density and center of mass
        pixel_density = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
        
        # 7. Calculate moments for orientation
        moments = cv2.moments(binary)
        orientation = 0
        if moments['m00'] != 0:
            # Second order moments for orientation
            orientation_numerator = 2 * moments['mu11']
            orientation_denominator = moments['mu20'] - moments['mu02']
            
            if orientation_denominator != 0:
                orientation = 0.5 * np.arctan2(orientation_numerator, orientation_denominator)
        
        # 8. Calculate the distribution of pixels along the x and y axes
        x_proj = np.sum(binary, axis=0) / max(np.sum(binary), 1)
        y_proj = np.sum(binary, axis=1) / max(np.sum(binary), 1)
        
        # Calculate skewness of projections (indicates slant)
        x_center = np.sum(np.arange(len(x_proj)) * x_proj) / max(np.sum(x_proj), 1)
        x_skew = np.sum(((np.arange(len(x_proj)) - x_center) ** 3) * x_proj) / max(np.sum(x_proj), 1)
        
        # 9. Create character-specific feature vector
        character_features = np.array([
            aspect_ratio,
            pixel_density,
            diag_ratio,
            diag_length_ratio,
            avg_slant,
            abs(orientation),
            np.sin(2 * abs(orientation)),
            x_skew / 1000.0,  # Scaled to be in reasonable range
            np.std(x_proj),
            np.std(y_proj),
            pos_diag_count / max(total_lines, 1),
            neg_diag_count / max(total_lines, 1)
        ])
        
        # Combine histogram and other features
        return np.concatenate([hist, character_features])
    
    def extract_word_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a word image using the original word-level extractor.
        
        This is useful for comparing with character-level results.
        
        Args:
            image: Grayscale image of a word
            
        Returns:
            Feature vector for the word
        """
        return self.word_feature_extractor.extract_features(image)
    
    def extract_character_features_from_file(self, image_path: str) -> np.ndarray:
        """Extract features from a character image file.
        
        Args:
            image_path: Path to character image
            
        Returns:
            Feature vector for the character
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read character image: {image_path}")
        
        return self.extract_character_features(image)
    
    def aggregate_character_features(self, character_features: List[np.ndarray]) -> np.ndarray:
        """Aggregate features from multiple characters to create a word-level representation.
        
        Args:
            character_features: List of feature vectors for each character
            
        Returns:
            Aggregated feature vector for the word
        """
        if not character_features:
            raise ValueError("No character features provided for aggregation")
        
        # Convert list to array
        features_array = np.vstack(character_features)
        
        # Calculate statistics across characters
        mean_features = np.mean(features_array, axis=0)
        std_features = np.std(features_array, axis=0)
        max_features = np.max(features_array, axis=0)
        
        # Combine statistics into a single feature vector
        return np.concatenate([mean_features, std_features, max_features])