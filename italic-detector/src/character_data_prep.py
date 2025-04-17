import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import cv2
import tempfile
from sklearn.model_selection import train_test_split

from .character_segmentation import CharacterSegmentation
from .character_feature_extractor import CharacterFeatureExtractor
from .data_prep import DataPreparation

class CharacterDataPreparation:
    """Prepares character-level data for training the italic detection model."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 processed_dir: str = "data/processed/character_level"):
        """Initialize character data preparation.
        
        Args:
            data_dir: Base directory with italic and regular text images
            processed_dir: Directory to save processed data
        """
        self.data_dir = data_dir
        self.italic_dir = os.path.join(data_dir, "italic")
        self.regular_dir = os.path.join(data_dir, "regular")
        self.processed_dir = processed_dir
        
        # Create directories if they don't exist
        os.makedirs(self.italic_dir, exist_ok=True)
        os.makedirs(self.regular_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize segmentation and feature extraction
        self.segmenter = CharacterSegmentation()
        self.feature_extractor = CharacterFeatureExtractor()
        
        # Base data preparation for word-level features
        self.word_data_prep = DataPreparation(data_dir, processed_dir)
    
    def collect_image_paths(self) -> Tuple[List[str], List[str]]:
        """Collect paths of italic and regular text images.
        
        Returns:
            Tuple of (italic_paths, regular_paths)
        """
        return self.word_data_prep.collect_image_paths()
    
    def segment_and_extract_features(self, 
                                    image_paths: List[str], 
                                    label: int,
                                    save_chars: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Segment images into characters and extract features.
        
        Args:
            image_paths: List of paths to text images
            label: Label for these images (0 for regular, 1 for italic)
            save_chars: Whether to save segmented characters to disk
            
        Returns:
            Tuple of (character_features, word_features):
                - character_features: List of feature vectors for each character
                - word_features: List of feature vectors for each word
        """
        character_features = []
        word_features = []
        
        # Create temp directory for saving character images if needed
        char_dir = None
        if save_chars:
            char_dir = os.path.join(self.processed_dir, "characters", "italic" if label == 1 else "regular")
            os.makedirs(char_dir, exist_ok=True)
            
        # Process each image
        for i, image_path in enumerate(tqdm(image_paths, desc=f"Processing {'italic' if label == 1 else 'regular'} images")):
            try:
                # Read image
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Error reading image: {image_path}")
                    continue
                
                # Extract word-level features
                word_feature = self.feature_extractor.extract_word_features(image)
                word_features.append(word_feature)
                
                # Segment into characters
                characters = self.segmenter.segment_characters(image)
                
                # Skip if no characters detected
                if not characters:
                    print(f"No characters detected in {image_path}")
                    continue
                
                # Save characters if requested
                if save_chars and char_dir:
                    word_char_dir = os.path.join(char_dir, f"word_{i}")
                    os.makedirs(word_char_dir, exist_ok=True)
                    
                    for j, char in enumerate(characters):
                        char_path = os.path.join(word_char_dir, f"char_{j}.png")
                        cv2.imwrite(char_path, char['image'])
                
                # Extract features for each character
                for char in characters:
                    char_feature = self.feature_extractor.extract_character_features(char['image'])
                    character_features.append((char_feature, label))
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return character_features, word_features
    
    def create_dataset(self, test_size: float = 0.2, save_chars: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Create character and word datasets for training and testing.
        
        Args:
            test_size: Proportion of data to use for testing
            save_chars: Whether to save individual character images
            
        Returns:
            Dictionary with datasets:
                - 'character': Dict with 'train' and 'test' DataFrames
                - 'word': Dict with 'train' and 'test' DataFrames
        """
        print("Collecting image paths...")
        italic_paths, regular_paths = self.collect_image_paths()
        
        if not italic_paths:
            raise ValueError(f"No italic images found in {self.italic_dir}")
        if not regular_paths:
            raise ValueError(f"No regular images found in {self.regular_dir}")
            
        print(f"Found {len(italic_paths)} italic images and {len(regular_paths)} regular images")
        
        # Process italic images
        print("Segmenting and extracting features from italic images...")
        italic_char_features, italic_word_features = self.segment_and_extract_features(
            italic_paths, label=1, save_chars=save_chars
        )
        
        # Process regular images
        print("Segmenting and extracting features from regular images...")
        regular_char_features, regular_word_features = self.segment_and_extract_features(
            regular_paths, label=0, save_chars=save_chars
        )
        
        # Prepare character-level dataset
        char_features = []
        char_labels = []
        
        for feature, label in italic_char_features + regular_char_features:
            char_features.append(feature)
            char_labels.append(label)
        
        X_char = np.vstack(char_features)
        y_char = np.array(char_labels)
        
        # Create character-level DataFrame
        char_feature_cols = [f"char_feature_{i}" for i in range(X_char.shape[1])]
        char_df = pd.DataFrame(X_char, columns=char_feature_cols)
        char_df['label'] = y_char
        
        # Split character data
        char_train_df, char_test_df = train_test_split(
            char_df, test_size=test_size, random_state=42, stratify=y_char
        )
        
        # Save character datasets
        char_train_df.to_csv(os.path.join(self.processed_dir, "char_train_data.csv"), index=False)
        char_test_df.to_csv(os.path.join(self.processed_dir, "char_test_data.csv"), index=False)
        
        print(f"Created character dataset with {len(char_train_df)} training samples and {len(char_test_df)} test samples")
        
        # Prepare word-level dataset with aggregated character features
        word_features = []
        word_labels = []
        
        # Aggregate character features for each word
        char_features_by_word = {}
        for i, path in enumerate(italic_paths):
            char_features_by_word[i] = []
        
        for i, path in enumerate(regular_paths):
            char_features_by_word[i + len(italic_paths)] = []
        
        # Aggregate word features
        all_word_features = italic_word_features + regular_word_features
        all_word_labels = [1] * len(italic_word_features) + [0] * len(regular_word_features)
        
        X_word = np.vstack(all_word_features)
        y_word = np.array(all_word_labels)
        
        # Create word-level DataFrame
        word_feature_cols = [f"word_feature_{i}" for i in range(X_word.shape[1])]
        word_df = pd.DataFrame(X_word, columns=word_feature_cols)
        word_df['label'] = y_word
        
        # Split word data
        word_train_df, word_test_df = train_test_split(
            word_df, test_size=test_size, random_state=42, stratify=y_word
        )
        
        # Save word datasets
        word_train_df.to_csv(os.path.join(self.processed_dir, "word_train_data.csv"), index=False)
        word_test_df.to_csv(os.path.join(self.processed_dir, "word_test_data.csv"), index=False)
        
        print(f"Created word dataset with {len(word_train_df)} training samples and {len(word_test_df)} test samples")
        
        return {
            'character': {
                'train': char_train_df,
                'test': char_test_df
            },
            'word': {
                'train': word_train_df,
                'test': word_test_df
            }
        }