# Character-Level Italic Text Detection

This is an extension of the original italic-detector with character-level detection capabilities. It provides more fine-grained analysis by segmenting words into individual characters and detecting italic styling at the character level.

## Features

- **Character Segmentation**: Splits text images into individual characters for fine-grained analysis
- **Character-Level Feature Extraction**: Extracts features specific to individual characters
- **Multi-Level Classification**: Provides both character-level and word-level predictions
- **Visualization**: Visualizes character segmentation and predictions for better understanding

## Usage

### Preparing Character-Level Data

```bash
# Prepare character-level dataset (segmenting words into characters)
char-detector prepare-character-data

# Save individual character images for inspection
char-detector prepare-character-data --save-chars
```

### Training Character-Level Models

```bash
# Train both character-level and word-level models
char-detector train-character-model
```

### Detecting Italic Text in an Image

```bash
# Analyze text image with character-level detection
char-detector detect-italic-chars "path/to/image.png"

# Show visualization of character segmentation and detection
char-detector detect-italic-chars "path/to/image.png" --visualize

# Adjust confidence threshold for character-level detection
char-detector detect-italic-chars "path/to/image.png" --char-threshold 0.65
```

### Processing Documents

```bash
# Process a document with character-level detection
char-detector analyze-document-chars "path/to/document.pdf"
```

## Character-Level Approach

The character-level approach offers several advantages:

1. **Fine-Grained Analysis**: Can detect italic styling on specific characters or portions of text
2. **Better Handling of Mixed Styling**: Works well with text that contains both italic and regular characters
3. **Improved Context**: Uses character position information to improve detection
4. **Visual Explanations**: Creates visualizations showing which characters are detected as italic

## Technical Details

The system works through these steps:

1. **Character Segmentation**: Uses contour detection to separate characters
2. **Feature Extraction**: Extracts features such as:
   - Character aspect ratio
   - Stroke angle distribution
   - Diagonal vs. vertical line ratios
   - Pixel density and distribution
   - Moment-based orientation
3. **Multi-Level Classification**: Uses both:
   - Character-level model trained on individual characters
   - Word-level model trained on aggregated character features
4. **Decision Fusion**: Combines character and word level predictions for final decision

## Examples

Example visualization of character-level detection:

```
I R R I I I I
```

Where:
- I: Characters classified as italic
- R: Characters classified as regular

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- Pandas
- PyMuPDF
- Click

## Installation

```bash
pip install -e .
```

## License

MIT