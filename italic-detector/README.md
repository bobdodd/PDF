# Italic Text Detector

A machine learning application to identify italic text in scanned documents. This application can be integrated with OCR systems to determine text styling.

## Features

- Extract text samples from PDF documents
- Train a machine learning model to identify italic vs regular text
- Deploy the model to Ollama for seamless integration with LLM applications
- Process full documents to detect italic text regions
- Easy to use command-line interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd italic-detector
```

2. Install dependencies:
```bash
pip install -e .
```

3. Make sure you have installed Ollama for model deployment:
   - Visit [Ollama's website](https://ollama.ai) for installation instructions

## Directory Structure

- `data/`: Contains training data and processed datasets
  - `italic/`: Place italic text images here
  - `regular/`: Place regular text images here
  - `spare_regular/`: Additional regular text samples (not used for training)
  - `processed/`: Processed data files used for training
- `models/`: Contains saved model files
- `src/`: Source code for the application

## Usage

### Extracting and Labeling Training Data

1. **Extract text samples from PDFs**:
```bash
italic-detector extract-pdf-data --pdf-folder /path/to/pdfs --output-dir data/extracted
```

2. **Label the extracted samples**:
```bash
italic-detector label-samples --source-dir data/extracted
```
   - You'll be shown each image and prompted to label it as italic (1) or regular (2)
   - Labeled images will be saved to `data/italic` and `data/spare_regular` respectively

### Prepare Dataset

Process the image data and create training/test datasets:
```bash
# Standard preparation
italic-detector prepare-data

# With data augmentation (increases dataset size through variations)
italic-detector prepare-data --augment
```

The `--augment` flag enables automatic data augmentation, which creates multiple variations of each image using:
- Small rotations (±1-3°)
- Extensive scaling (60-120%)
- Horizontal and vertical stretching (±5-10%)
- Minor brightness/contrast adjustments
- Controlled noise addition
- Combined transformations (e.g., rotation+scaling, rotation+stretching)
- Multi-level transformations (up to 3 transforms applied sequentially)

This can significantly improve model accuracy by providing more training examples and making the model more robust to variations in text appearance.

### Train Model

Train the italic detection model:
```bash
italic-detector train-model
```

### Deploy to Ollama

Deploy the trained model to Ollama for integration with LLMs:
```bash
italic-detector deploy-to-ollama --model-name italic-detector
```

### Detect Italic Text

Analyze a single text image:
```bash
italic-detector detect-italic --image-path text_sample.png
```

Process an entire document:
```bash
italic-detector process-document --pdf-path document.pdf
```

## Model Details

The model uses a combination of computer vision features to detect italic text:
- Angle histograms from Hough line detection
- Pixel density measurements
- Diagonal vs. vertical line ratios
- Image moment analysis for text orientation

## License

[MIT License](LICENSE)