import os
import pickle
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ItalicDetectionModel:
    """Model for detecting italic text in images."""
    
    def __init__(self, model_path: str = "models/saved"):
        """Initialize the model."""
        self.model_path = model_path
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on the given data."""
        print("Training model...")
        
        # Create a random forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        print("Model training complete")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if text is italic for given feature vectors."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get raw predictions (0 for regular, 1 for italic)
        raw_predictions = self.model.predict(X)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        # Apply a threshold for classifying as italic (balanced approach)
        # Only mark as italic if confidence is high enough
        italic_threshold = 0.54  # Use 54% confidence for better balance between precision and recall
        
        # Apply threshold
        adjusted_predictions = np.zeros_like(raw_predictions)
        for i, pred in enumerate(raw_predictions):
            if pred == 1 and probabilities[i, 1] >= italic_threshold:
                adjusted_predictions[i] = 1
        
        return adjusted_predictions
    
    def save(self, filename: str = "italic_detector.pkl") -> str:
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_file = os.path.join(self.model_path, filename)
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {model_file}")
        return model_file
    
    def load(self, filename: str = "italic_detector.pkl") -> None:
        """Load the model from a file."""
        model_file = os.path.join(self.model_path, filename)
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {model_file}")
    
    def export_to_onnx(self, filename: str = "italic_detector.onnx") -> str:
        """Export the model to ONNX format for Ollama."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        onnx_file = os.path.join(self.model_path, filename)
        
        # Convert model to ONNX
        initial_type = [('float_input', FloatTensorType([None, self.model.n_features_in_]))]
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        
        # Save ONNX model
        with open(onnx_file, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Model exported to ONNX format: {onnx_file}")
        return onnx_file