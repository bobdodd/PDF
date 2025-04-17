import numpy as np
import cv2
from typing import List, Tuple, Dict
import os

class CharacterSegmentation:
    """Segments text images into individual characters for finer-grained analysis."""
    
    def __init__(self, 
                 min_char_width: int = 3,
                 min_char_height: int = 5,
                 padding: int = 1):
        """Initialize the character segmentation with parameters.
        
        Args:
            min_char_width: Minimum width in pixels for a valid character
            min_char_height: Minimum height in pixels for a valid character
            padding: Number of pixels to pad around each character
        """
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height
        self.padding = padding
    
    def segment_characters(self, image: np.ndarray) -> List[Dict[str, any]]:
        """Segment an image into individual characters.
        
        Args:
            image: Grayscale image containing text
            
        Returns:
            List of dictionaries, each containing:
                - 'image': The character image
                - 'bbox': (x, y, w, h) bounding box coordinates
                - 'position': Character position in the word (left-to-right)
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out noise and very small components
            if w >= self.min_char_width and h >= self.min_char_height:
                bboxes.append((x, y, w, h))
        
        # Sort bounding boxes from left to right
        bboxes.sort(key=lambda bbox: bbox[0])
        
        # Extract character images
        characters = []
        for i, (x, y, w, h) in enumerate(bboxes):
            # Add padding to character
            x_min = max(0, x - self.padding)
            y_min = max(0, y - self.padding)
            x_max = min(image.shape[1], x + w + self.padding)
            y_max = min(image.shape[0], y + h + self.padding)
            
            # Extract character image
            char_img = image[y_min:y_max, x_min:x_max]
            
            characters.append({
                'image': char_img,
                'bbox': (x, y, w, h),
                'position': i
            })
        
        return characters
    
    def visualize_segmentation(self, image: np.ndarray, characters: List[Dict]) -> np.ndarray:
        """Create a visualization of the character segmentation.
        
        Args:
            image: Original image
            characters: List of character dictionaries from segment_characters()
            
        Returns:
            Image with character bounding boxes drawn
        """
        # Create a copy of the image for visualization
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw bounding boxes
        for i, char in enumerate(characters):
            x, y, w, h = char['bbox']
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Put character index
            cv2.putText(vis_image, str(i), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return vis_image
    
    def save_characters(self, characters: List[Dict], output_dir: str) -> List[str]:
        """Save individual character images to disk.
        
        Args:
            characters: List of character dictionaries from segment_characters()
            output_dir: Directory to save character images
            
        Returns:
            List of paths to saved character images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, char in enumerate(characters):
            # Create filename
            filename = f"char_{i}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, char['image'])
            saved_paths.append(filepath)
        
        return saved_paths