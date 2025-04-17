import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CharacterModel:
    """Model for detecting italic text at the character level."""
    
    def __init__(self, model_path: str = "models/saved/character_level"):
        """Initialize the character-level detection model.
        
        Args:
            model_path: Directory to save/load model files
        """
        self.model_path = model_path
        self.character_model = None
        self.word_level_model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
    
    def train_character_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the character-level model on the given data.
        
        Args:
            X_train: Feature vectors for characters
            y_train: Labels (0 for regular, 1 for italic)
        """
        print("Training character-level model...")
        
        # Create a random forest classifier
        self.character_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        # Train the model
        self.character_model.fit(X_train, y_train)
        
        print("Character model training complete")
    
    def train_word_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the word-level aggregation model on character features.
        
        Args:
            X_train: Aggregated character features for words
            y_train: Labels (0 for regular, 1 for italic)
        """
        print("Training word-level aggregation model...")
        
        # Create a random forest classifier
        self.word_level_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        # Train the model
        self.word_level_model.fit(X_train, y_train)
        
        print("Word-level model training complete")
    
    def predict_character(self, X: np.ndarray) -> np.ndarray:
        """Predict if characters are italic based on their features.
        
        Args:
            X: Feature vectors for characters
            
        Returns:
            Predictions (0 for regular, 1 for italic)
        """
        if self.character_model is None:
            raise ValueError("Character model not trained. Call train_character_model() first.")
        
        # Get probabilities
        probabilities = self.character_model.predict_proba(X)
        
        # Apply threshold for italic classification
        italic_threshold = 0.60  # Use 60% confidence for character level
        
        # Create predictions based on threshold
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            if probabilities[i, 1] >= italic_threshold:
                predictions[i] = 1
        
        return predictions
    
    def predict_word(self, X: np.ndarray) -> np.ndarray:
        """Predict if words are italic based on aggregated character features.
        
        Args:
            X: Aggregated feature vectors for words
            
        Returns:
            Predictions (0 for regular, 1 for italic)
        """
        if self.word_level_model is None:
            raise ValueError("Word model not trained. Call train_word_model() first.")
        
        # Get probabilities
        probabilities = self.word_level_model.predict_proba(X)
        
        # Apply threshold for italic classification
        italic_threshold = 0.60  # Use 60% confidence
        
        # Create predictions based on threshold
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            if probabilities[i, 1] >= italic_threshold:
                predictions[i] = 1
        
        return predictions
    
    def evaluate_character_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the character-level model on test data.
        
        Args:
            X_test: Feature vectors for test characters
            y_test: True labels for test characters
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.character_model is None:
            raise ValueError("Character model not trained. Call train_character_model() first.")
        
        # Make predictions
        y_pred = self.predict_character(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Character-level model evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_word_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the word-level model on test data.
        
        Args:
            X_test: Aggregated feature vectors for test words
            y_test: True labels for test words
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.word_level_model is None:
            raise ValueError("Word model not trained. Call train_word_model() first.")
        
        # Make predictions
        y_pred = self.predict_word(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Word-level model evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_models(self) -> Tuple[str, str]:
        """Save the character and word models to files.
        
        Returns:
            Tuple of (character_model_path, word_model_path)
        """
        if self.character_model is None:
            raise ValueError("Character model not trained. Call train_character_model() first.")
        
        char_model_file = os.path.join(self.model_path, "character_model.pkl")
        
        with open(char_model_file, 'wb') as f:
            pickle.dump(self.character_model, f)
        
        print(f"Character model saved to {char_model_file}")
        
        # Save word model if it exists
        word_model_file = None
        if self.word_level_model is not None:
            word_model_file = os.path.join(self.model_path, "word_model.pkl")
            
            with open(word_model_file, 'wb') as f:
                pickle.dump(self.word_level_model, f)
            
            print(f"Word model saved to {word_model_file}")
        
        return char_model_file, word_model_file
    
    def load_models(self, char_model_file: str = None, word_model_file: str = None) -> None:
        """Load the character and word models from files.
        
        Args:
            char_model_file: Path to character model file
            word_model_file: Path to word model file
        """
        # Load character model
        if char_model_file is None:
            char_model_file = os.path.join(self.model_path, "character_model.pkl")
        
        if os.path.exists(char_model_file):
            with open(char_model_file, 'rb') as f:
                self.character_model = pickle.load(f)
            
            print(f"Character model loaded from {char_model_file}")
        
        # Load word model
        if word_model_file is None:
            word_model_file = os.path.join(self.model_path, "word_model.pkl")
        
        if os.path.exists(word_model_file):
            with open(word_model_file, 'rb') as f:
                self.word_level_model = pickle.load(f)
            
            print(f"Word model loaded from {word_model_file}")
    
    def export_to_onnx(self) -> Tuple[str, str]:
        """Export the models to ONNX format.
        
        Returns:
            Tuple of (character_model_path, word_model_path)
        """
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        char_onnx_file = None
        word_onnx_file = None
        
        # Export character model
        if self.character_model is not None:
            char_onnx_file = os.path.join(self.model_path, "character_model.onnx")
            
            # Convert model to ONNX
            initial_type = [('float_input', FloatTensorType([None, self.character_model.n_features_in_]))]
            onnx_model = convert_sklearn(self.character_model, initial_types=initial_type)
            
            # Save ONNX model
            with open(char_onnx_file, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"Character model exported to ONNX format: {char_onnx_file}")
        
        # Export word model
        if self.word_level_model is not None:
            word_onnx_file = os.path.join(self.model_path, "word_model.onnx")
            
            # Convert model to ONNX
            initial_type = [('float_input', FloatTensorType([None, self.word_level_model.n_features_in_]))]
            onnx_model = convert_sklearn(self.word_level_model, initial_types=initial_type)
            
            # Save ONNX model
            with open(word_onnx_file, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"Word model exported to ONNX format: {word_onnx_file}")
        
        return char_onnx_file, word_onnx_file