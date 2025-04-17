import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import fitz  # PyMuPDF
from .feature_extractor import FeatureExtractor

class DataPreparation:
    """Prepares data for training the italic text detector model."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 processed_dir: str = "data/processed"):
        """Initialize data preparation with directories."""
        self.data_dir = data_dir
        self.italic_dir = os.path.join(data_dir, "italic")
        self.regular_dir = os.path.join(data_dir, "regular")
        self.spare_regular_dir = os.path.join(data_dir, "spare_regular")
        self.processed_dir = processed_dir
        
        # Create directories if they don't exist
        os.makedirs(self.italic_dir, exist_ok=True)
        os.makedirs(self.regular_dir, exist_ok=True)
        os.makedirs(self.spare_regular_dir, exist_ok=True)
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
    
    def collect_image_paths(self) -> Tuple[List[str], List[str]]:
        """Collect paths of italic and regular text images."""
        # Collect italic images
        italic_paths = []
        for file in os.listdir(self.italic_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                italic_paths.append(os.path.join(self.italic_dir, file))
        
        # Collect regular images
        regular_paths = []
        for file in os.listdir(self.regular_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                regular_paths.append(os.path.join(self.regular_dir, file))
        
        return italic_paths, regular_paths
    
    def apply_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation to create variations of the input image."""
        augmented_images = []
        
        # Original image is always included
        augmented_images.append(image)
        
        # Define transformation parameters
        angles = [-3, -2, -1, 1, 2, 3]
        scales = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2]
        stretch_factors = [0.9, 0.95, 1.05, 1.1]
        noise_sigmas = [5, 10]
        contrast_factors = [0.9, 1.1]
        brightness_shifts = [-10, 10]
        
        # Helper functions for each transformation
        def apply_rotation(img, angle):
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        def apply_scaling(img, scale):
            h, w = img.shape[:2]
            scaled_w, scaled_h = int(w * scale), int(h * scale)
            scaled = cv2.resize(img, (scaled_w, scaled_h))
            
            result = np.ones((h, w), dtype=np.uint8) * 255
            offset_x = max(0, (w - scaled_w) // 2)
            offset_y = max(0, (h - scaled_h) // 2)
            src_x, src_y = max(0, (scaled_w - w) // 2), max(0, (scaled_h - h) // 2)
            src_w = min(scaled_w, w)
            src_h = min(scaled_h, h)
            
            result[offset_y:offset_y+src_h, offset_x:offset_x+src_w] = \
                scaled[src_y:src_y+src_h, src_x:src_x+src_w]
            return result
        
        def apply_h_stretch(img, factor):
            h, w = img.shape[:2]
            stretched_w = int(w * factor)
            stretched_h = h
            stretched = cv2.resize(img, (stretched_w, stretched_h))
            
            result = np.ones((h, w), dtype=np.uint8) * 255
            if stretched_w <= w:
                offset_x = (w - stretched_w) // 2
                result[:, offset_x:offset_x+stretched_w] = stretched
            else:
                crop_start = (stretched_w - w) // 2
                result[:, :] = stretched[:, crop_start:crop_start+w]
            return result
        
        def apply_v_stretch(img, factor):
            h, w = img.shape[:2]
            stretched_h = int(h * factor)
            stretched_w = w
            stretched = cv2.resize(img, (stretched_w, stretched_h))
            
            result = np.ones((h, w), dtype=np.uint8) * 255
            if stretched_h <= h:
                offset_y = (h - stretched_h) // 2
                result[offset_y:offset_y+stretched_h, :] = stretched
            else:
                crop_start = (stretched_h - h) // 2
                result[:, :] = stretched[crop_start:crop_start+h, :]
            return result
        
        def apply_noise(img, sigma):
            noisy = img.copy()
            gaussian = np.random.normal(0, sigma, img.shape).astype(np.uint8)
            return cv2.add(noisy, gaussian)
        
        def apply_brightness_contrast(img, alpha, beta):
            return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 1. Basic single transformations
        # Rotations
        for angle in angles:
            augmented_images.append(apply_rotation(image, angle))
        
        # Scalings
        for scale in scales:
            augmented_images.append(apply_scaling(image, scale))
        
        # Horizontal and vertical stretching
        for factor in stretch_factors:
            augmented_images.append(apply_h_stretch(image, factor))
            augmented_images.append(apply_v_stretch(image, factor))
        
        # Noise
        for sigma in noise_sigmas:
            augmented_images.append(apply_noise(image, sigma))
        
        # Brightness/contrast
        for alpha in contrast_factors:
            for beta in brightness_shifts:
                augmented_images.append(apply_brightness_contrast(image, alpha, beta))
        
        # 2. Combined transformations
        
        # Rotation + scaling (select combinations to avoid explosion)
        for angle in [-2, 2]:
            for scale in [0.8, 1.2]:
                rotated = apply_rotation(image, angle)
                augmented_images.append(apply_scaling(rotated, scale))
        
        # Rotation + horizontal stretch
        for angle in [-2, 2]:
            for factor in [0.9, 1.1]:
                rotated = apply_rotation(image, angle)
                augmented_images.append(apply_h_stretch(rotated, factor))
        
        # Rotation + vertical stretch
        for angle in [-2, 2]:
            for factor in [0.9, 1.1]:
                rotated = apply_rotation(image, angle)
                augmented_images.append(apply_v_stretch(rotated, factor))
        
        # Scaling + stretching
        for scale in [0.8, 1.2]:
            for factor in [0.9, 1.1]:
                scaled = apply_scaling(image, scale)
                augmented_images.append(apply_h_stretch(scaled, factor))
                augmented_images.append(apply_v_stretch(scaled, factor))
        
        # Brightness/contrast + rotation
        for alpha in contrast_factors:
            for angle in [-2, 2]:
                adjusted = apply_brightness_contrast(image, alpha, 0)
                augmented_images.append(apply_rotation(adjusted, angle))
        
        # Triple combinations (very selective to avoid explosion)
        # Rotation + scaling + horizontal stretch
        rotated = apply_rotation(image, 2)
        scaled = apply_scaling(rotated, 0.9)
        augmented_images.append(apply_h_stretch(scaled, 1.1))
        
        rotated = apply_rotation(image, -2)
        scaled = apply_scaling(rotated, 1.1)
        augmented_images.append(apply_h_stretch(scaled, 0.9))
        
        return augmented_images
    
    def create_dataset(self, test_size: float = 0.2, augment: bool = False) -> Dict[str, pd.DataFrame]:
        """Create and split the dataset for training and testing."""
        print("Collecting image paths...")
        italic_paths, regular_paths = self.collect_image_paths()
        
        if not italic_paths:
            raise ValueError(f"No italic images found in {self.italic_dir}")
        if not regular_paths:
            raise ValueError(f"No regular images found in {self.regular_dir}")
            
        print(f"Found {len(italic_paths)} italic images and {len(regular_paths)} regular images")
        
        # Extract features for all images
        print("Extracting features from italic images...")
        italic_features = []
        italic_augmented_count = 0
        
        for path in tqdm(italic_paths):
            try:
                if not augment:
                    # Standard processing without augmentation
                    features = self.feature_extractor.extract_features_from_file(path)
                    italic_features.append(features)
                else:
                    # With augmentation
                    # First read the image
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error reading image: {path}")
                        continue
                        
                    # Apply augmentation to get multiple variations
                    augmented_images = self.apply_augmentation(image)
                    
                    # Extract features from each augmented version
                    for aug_img in augmented_images:
                        features = self.feature_extractor.extract_features(aug_img)
                        italic_features.append(features)
                        italic_augmented_count += 1
                        
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        print("Extracting features from regular images...")
        regular_features = []
        regular_augmented_count = 0
        
        for path in tqdm(regular_paths):
            try:
                if not augment:
                    # Standard processing without augmentation
                    features = self.feature_extractor.extract_features_from_file(path)
                    regular_features.append(features)
                else:
                    # With augmentation
                    # First read the image
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        print(f"Error reading image: {path}")
                        continue
                        
                    # Apply augmentation to get multiple variations
                    augmented_images = self.apply_augmentation(image)
                    
                    # Extract features from each augmented version
                    for aug_img in augmented_images:
                        features = self.feature_extractor.extract_features(aug_img)
                        regular_features.append(features)
                        regular_augmented_count += 1
                        
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        # Create labels
        italic_labels = np.ones(len(italic_features))
        regular_labels = np.zeros(len(regular_features))
        
        # Combine features and labels
        X = np.vstack((italic_features, regular_features))
        y = np.hstack((italic_labels, regular_labels))
        
        # Create DataFrame
        feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_cols)
        df['label'] = y
        
        # Split into training and testing sets
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=y
        )
        
        # Print augmentation summary if enabled
        if augment:
            orig_italic = len(italic_paths)
            orig_regular = len(regular_paths)
            print(f"\nData augmentation summary:")
            print(f"  Original italic samples: {orig_italic}")
            print(f"  Augmented italic samples: {italic_augmented_count}")
            print(f"  Multiplication factor: {italic_augmented_count / max(1, orig_italic):.1f}x")
            print(f"  Original regular samples: {orig_regular}")
            print(f"  Augmented regular samples: {regular_augmented_count}")
            print(f"  Multiplication factor: {regular_augmented_count / max(1, orig_regular):.1f}x")
        
        print(f"Created dataset with {len(train_df)} training samples and {len(test_df)} test samples")
        
        # Save datasets
        train_df.to_csv(os.path.join(self.processed_dir, "train_data.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test_data.csv"), index=False)
        
        return {
            'train': train_df,
            'test': test_df
        }
    
    @staticmethod
    def extract_text_samples_from_pdf(pdf_path: str, output_dir: str, min_size: int = 15):
        """Extract text samples from a PDF for creating a dataset."""
        doc = fitz.open(pdf_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        sample_count = 0
        
        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            
            # Convert to grayscale
            if pix.n == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            else:  # RGB
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours (text regions)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small regions
                if h < min_size or w < 5:
                    continue
                
                # Extract the word region
                word_img = gray[y:y+h, x:x+w]
                
                # Save the word image
                output_path = os.path.join(output_dir, f"page_{page_num+1}_word_{i}.png")
                cv2.imwrite(output_path, word_img)
                sample_count += 1
        
        print(f"Extracted {sample_count} text samples from {pdf_path} to {output_dir}")
        
    @staticmethod
    def process_pdf_folder(pdf_folder: str, output_dir: str, temp_dir: str = None, min_size: int = 15):
        """Process multiple PDFs from a folder to extract text samples."""
        # Create temp directory if provided and it doesn't exist
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {pdf_folder}")
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        total_samples = 0
        
        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            
            # Create PDF-specific output directory
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            os.makedirs(pdf_output_dir, exist_ok=True)
            
            # Process the PDF
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Create page directory
                page_dir = os.path.join(pdf_output_dir, f"page_{page_num+1}")
                if temp_dir:
                    page_dir = os.path.join(temp_dir, f"{pdf_name}_page_{page_num+1}")
                os.makedirs(page_dir, exist_ok=True)
                
                # Render page to image
                pix = page.get_pixmap(dpi=300)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                # Save full page image
                page_img_path = os.path.join(pdf_output_dir, f"page_{page_num+1}.png")
                if pix.n == 4:  # RGBA
                    cv2.imwrite(page_img_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
                    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                else:  # RGB
                    cv2.imwrite(page_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Apply thresholding
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Find contours (text regions)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter out very small regions
                    if h < min_size or w < 5:
                        continue
                    
                    # Extract the word region
                    word_img = gray[y:y+h, x:x+w]
                    
                    # Save the word image
                    output_path = os.path.join(page_dir, f"word_{i}.png")
                    cv2.imwrite(output_path, word_img)
                    total_samples += 1
            
            doc.close()
            
        print(f"Extracted {total_samples} text samples from {len(pdf_files)} PDFs")
        return total_samples