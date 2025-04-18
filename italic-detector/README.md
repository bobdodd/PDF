# Character-Level Italic Text Detector

A sophisticated machine learning system for detecting italic text in images using both character-level and word-level analysis, building on the base functionality of the original italic-detector.

## Overview

The Character-Level Italic Text Detector combines two approaches for detecting italic text in images:

1. **Character-Level Detection**: Segments text into individual characters and analyzes each character's features to determine if it's italic.
2. **Word-Level Detection**: Analyzes features of entire words to determine if they're italic.

By combining these approaches, the system achieves high accuracy across different fonts and text styles.

## Features

- **Dual-Level Analysis**: Uses both character and word-level models for robust detection
- **Intelligent Override Logic**: Automatically resolves conflicts between character and word-level predictions
- **Visualization**: Generates annotated images showing detection results for each character
- **Ollama Integration**: Easy deployment to Ollama for inference via REST API
- **Flexible Deployment**: Works with local models or Ollama running on any host/port
- **Data Augmentation**: Increases dataset size through variations (rotations, scaling, etc.)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/italic-detector.git
cd italic-detector

# Install dependencies
pip install -e .
```

## Usage

### Preparing Data

Process the image data and create training/test datasets:
```bash
# Standard preparation for word-level detection
italic-detector prepare-data

# With data augmentation
italic-detector prepare-data --augment

# Prepare character-level dataset (segmenting words into characters)
char-detector prepare-character-data

# Save individual character images for inspection
char-detector prepare-character-data --save-chars
```

The `--augment` flag enables automatic data augmentation, which creates multiple variations of each image using:
- Small rotations (±1-3°)
- Extensive scaling (60-120%)
- Horizontal and vertical stretching (±5-10%)
- Minor brightness/contrast adjustments
- Controlled noise addition

### Training Models

```bash
# Train the word-level model
italic-detector train-model

# Train the character-level model
char-detector train-character-model

# Train with data augmentation
char-detector train-character-model --augment
```

### Detecting Italic Text

```bash
# Word-level detection
italic-detector detect-italic --image-path text_sample.png

# Character-level detection
char-detector detect-italic-chars text_sample.png

# Character-level detection with visualization
char-detector detect-italic-chars text_sample.png --visualize
```

### Using the Ollama API Client

The package includes a standalone Ollama API client for easier integration and inference:

```bash
# Use local models
python examples/ollama_api_client.py image_path.png --use-local

# Use Ollama with visualization saved to /tmp
python examples/ollama_api_client.py image_path.png --visualize

# Use Ollama with custom model name
python examples/ollama_api_client.py image_path.png --model your-model-name

# Connect to Ollama on a different host
python examples/ollama_api_client.py image_path.png --api-url http://192.168.1.100:11434/api

# Save visualizations to a specific directory
python examples/ollama_api_client.py image_path.png --visualize --visualization-dir ./output
```

### Deploying to Ollama

```bash
# Deploy the word-level model to Ollama
italic-detector deploy-to-ollama --model-name italic-detector

# Deploy the character-level model to Ollama
char-detector deploy-to-ollama --model-name char-italic-detector
```

## How It Works

The detection process works as follows:

1. **Character Segmentation**: The system segments text into individual characters
2. **Feature Extraction**: Extracts features from both characters and words
3. **Classification**: Both character and word-level models predict if the text is italic
4. **Decision Logic**: A sophisticated algorithm resolves conflicts between the models:
   - High confidence (>80%) character-level detection overrides word-level
   - High confidence (>0.8) word-level prediction is trusted for italic text
   - Regular text detection (≤40% italic characters) overrides word-level unless highly confident
   - Moderate evidence (>50% italic characters) with low word-level confidence is classified as italic

## Recent Improvements

Recent enhancements to the system include:

1. **Character-Level Detection**: Fine-grained analysis of individual characters
2. **Improved Decision Logic**: Better handling of edge cases where character and word-level models disagree
3. **Ollama API Integration**: New REST API client for easy integration and deployment
4. **Flexible Visualization Options**: Configure visualization output to /tmp or any directory
5. **Remote Ollama Support**: Connect to Ollama instances running on different hosts

## Project Structure

```
italic-detector/
├── src/
│   ├── main.py                    # Main CLI interface for word-level detection
│   ├── model.py                   # Word-level model implementation
│   ├── feature_extractor.py       # Word-level feature extraction
│   ├── data_prep.py               # Word-level data preparation
│   ├── character_segmentation.py  # Character segmentation logic
│   ├── character_feature_extractor.py  # Feature extraction for characters
│   ├── character_model.py         # ML models for character/word classification
│   ├── character_data_prep.py     # Data preparation and augmentation
│   ├── character_main.py          # CLI commands for character-level functions
│   └── ollama_integration.py      # Ollama deployment utilities
├── examples/
│   ├── character_demo.py          # Simple demo script
│   └── ollama_api_client.py       # Advanced Ollama API client
├── data/
│   ├── italic/                    # Italic text samples
│   └── regular/                   # Regular text samples
└── models/                        # Saved ML models
    └── saved/
        └── character_level/       # Character-level models
```

## Technical Details

The model uses a combination of computer vision features to detect italic text:

### Word-Level Features
- Angle histograms from Hough line detection
- Pixel density measurements
- Diagonal vs. vertical line ratios
- Image moment analysis for text orientation

### Character-Level Features
- Character aspect ratio
- Stroke angle distribution
- Contour analysis
- Slant measurements

### Decision Fusion

The decision fusion process intelligently combines character and word-level predictions to handle challenging cases:
- Fonts where character-level detection might struggle
- Cases where word-level context is important
- Mixed italic/regular text

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- Pandas
- PyMuPDF
- Click
- Requests (for Ollama API)
- Matplotlib

## License

MIT