# API Reference

This document explains the programmatic API for the italic-detector library.

## FeatureExtractor

The FeatureExtractor class is responsible for extracting features from text images.

```python
from src.feature_extractor import FeatureExtractor

# Initialize feature extractor
extractor = FeatureExtractor(resize_shape=(100, 32), angle_bins=18)

# Extract features from an image file
features = extractor.extract_features_from_file("text_image.png")

# Extract features from an image in memory
import cv2
image = cv2.imread("text_image.png", cv2.IMREAD_GRAYSCALE)
features = extractor.extract_features(image)
```

### Key Methods

- `extract_features(image)`: Extracts features from a numpy array image
- `extract_features_from_file(image_path)`: Extracts features from an image file

## DataPreparation

The DataPreparation class handles dataset creation and preparation.

```python
from src.data_prep import DataPreparation

# Initialize data preparation with directories
data_prep = DataPreparation(
    data_dir="data",
    processed_dir="data/processed"
)

# Create dataset from image files
dataset = data_prep.create_dataset(test_size=0.2)

# Extract text samples from PDF
DataPreparation.extract_text_samples_from_pdf(
    pdf_path="document.pdf",
    output_dir="data/samples"
)
```

### Key Methods

- `create_dataset(test_size)`: Creates and splits datasets for training and testing
- `collect_image_paths()`: Collects paths of italic and regular text images
- `extract_text_samples_from_pdf(pdf_path, output_dir)`: Static method to extract text samples from a PDF

## ItalicDetectionModel

The ItalicDetectionModel class provides model training, evaluation, and persistence.

```python
from src.model import ItalicDetectionModel
import numpy as np

# Initialize model
model = ItalicDetectionModel(model_path="models/saved")

# Train model
model.train(X_train, y_train)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']}")

# Make predictions
predictions = model.predict(X_test)

# Save and load model
model_file = model.save("my_model.pkl")
model.load("my_model.pkl")

# Export model to ONNX format
onnx_file = model.export_to_onnx("my_model.onnx")
```

### Key Methods

- `train(X_train, y_train)`: Trains the model on provided data
- `evaluate(X_test, y_test)`: Evaluates model performance on test data
- `predict(X)`: Makes predictions for given feature vectors
- `save(filename)`: Saves the model to a file
- `load(filename)`: Loads the model from a file
- `export_to_onnx(filename)`: Exports the model to ONNX format

## OllamaIntegration

The OllamaIntegration class provides integration with Ollama.

```python
from src.ollama_integration import OllamaIntegration

# Initialize Ollama integration
ollama = OllamaIntegration(
    model_name="italic-detector",
    base_model="llama2",
    onnx_model_path="models/saved/italic_detector.onnx"
)

# Create Modelfile
modelfile_path = ollama.create_modelfile("Modelfile")

# Build model
success = ollama.build_model()

# Run prediction
result = ollama.run_prediction(features)
is_italic = result["is_italic"]
confidence = result["confidence"]
```

### Key Methods

- `create_modelfile(modelfile_path)`: Creates a Modelfile for Ollama
- `build_model(modelfile_path)`: Builds the Ollama model
- `run_prediction(features)`: Runs a prediction using the Ollama model