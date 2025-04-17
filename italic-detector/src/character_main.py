import os
import sys
import click
import pandas as pd
import numpy as np
import tempfile
from tqdm import tqdm
import cv2
import json

# Fix imports to work both as package and as script
try:
    from .character_segmentation import CharacterSegmentation
    from .character_feature_extractor import CharacterFeatureExtractor
    from .character_data_prep import CharacterDataPreparation
    from .character_model import CharacterModel
except ImportError:
    from character_segmentation import CharacterSegmentation
    from character_feature_extractor import CharacterFeatureExtractor
    from character_data_prep import CharacterDataPreparation
    from character_model import CharacterModel

@click.group()
def cli():
    """Character-Level Italic Text Detector."""
    pass

@cli.command()
@click.option('--test-size', type=float, default=0.2, 
              help='Proportion of data for testing')
@click.option('--save-chars', is_flag=True, 
              help='Save individual character images')
def prepare_character_data(test_size, save_chars):
    """Prepare character-level dataset for training."""
    data_prep = CharacterDataPreparation()
    
    if save_chars:
        click.echo("Character image saving enabled - character images will be saved to disk")
    
    dataset = data_prep.create_dataset(test_size=test_size, save_chars=save_chars)
    
    click.echo(f"Character dataset prepared with {len(dataset['character']['train'])} training samples "
             f"and {len(dataset['character']['test'])} test samples")
    click.echo(f"Word dataset prepared with {len(dataset['word']['train'])} training samples "
             f"and {len(dataset['word']['test'])} test samples")

@cli.command()
def train_character_model():
    """Train the character-level detection model."""
    # Check if processed data exists
    char_train_path = "data/processed/character_level/char_train_data.csv"
    char_test_path = "data/processed/character_level/char_test_data.csv"
    
    if not os.path.exists(char_train_path) or not os.path.exists(char_test_path):
        click.echo("Processed character data not found. Run 'prepare-character-data' first.")
        return
    
    # Load data
    click.echo("Loading character data...")
    char_train_df = pd.read_csv(char_train_path)
    char_test_df = pd.read_csv(char_test_path)
    
    # Separate features and labels
    X_char_train = char_train_df.drop('label', axis=1).values
    y_char_train = char_train_df['label'].values
    
    X_char_test = char_test_df.drop('label', axis=1).values
    y_char_test = char_test_df['label'].values
    
    # Train character model
    model = CharacterModel()
    model.train_character_model(X_char_train, y_char_train)
    
    # Evaluate character model
    metrics = model.evaluate_character_model(X_char_test, y_char_test)
    
    # Check if word data exists
    word_train_path = "data/processed/character_level/word_train_data.csv"
    word_test_path = "data/processed/character_level/word_test_data.csv"
    
    if os.path.exists(word_train_path) and os.path.exists(word_test_path):
        click.echo("Loading word data...")
        word_train_df = pd.read_csv(word_train_path)
        word_test_df = pd.read_csv(word_test_path)
        
        # Separate features and labels
        X_word_train = word_train_df.drop('label', axis=1).values
        y_word_train = word_train_df['label'].values
        
        X_word_test = word_test_df.drop('label', axis=1).values
        y_word_test = word_test_df['label'].values
        
        # Train word model
        model.train_word_model(X_word_train, y_word_train)
        
        # Evaluate word model
        word_metrics = model.evaluate_word_model(X_word_test, y_word_test)
    
    # Save models
    char_model_file, word_model_file = model.save_models()
    click.echo(f"Character model saved to {char_model_file}")
    if word_model_file:
        click.echo(f"Word model saved to {word_model_file}")
    
    # Export to ONNX
    char_onnx_file, word_onnx_file = model.export_to_onnx()
    if char_onnx_file:
        click.echo(f"Character model exported to ONNX format: {char_onnx_file}")
    if word_onnx_file:
        click.echo(f"Word model exported to ONNX format: {word_onnx_file}")

@cli.command()
@click.argument('image_path_arg', required=False)
@click.option('--image-path', type=click.Path(exists=True), 
              help='Path to text image to analyze')
@click.option('--visualize', is_flag=True, 
              help='Show character segmentation visualization')
@click.option('--char-threshold', type=float, default=0.60,
              help='Confidence threshold for character-level classification')
def detect_italic_chars(image_path_arg, image_path, visualize, char_threshold):
    """Detect italic text at character level.
    
    Can be called in two ways:
    
    italic-detector detect-italic-chars --image-path "path with spaces.png"
    
    OR simply:
    
    italic-detector detect-italic-chars "path with spaces.png"
    """
    # Determine which path to use
    if image_path_arg and os.path.exists(image_path_arg):
        actual_path = image_path_arg
    elif image_path:
        actual_path = image_path
    else:
        click.echo("Error: Please provide a valid image path either as an argument or with --image-path")
        return
    
    # Load model
    model = CharacterModel()
    try:
        model.load_models()
    except Exception as e:
        click.echo(f"Error loading models: {e}")
        click.echo("Please train the character model first with 'train-character-model'")
        return
    
    # Initialize segmentation and feature extraction
    segmenter = CharacterSegmentation()
    feature_extractor = CharacterFeatureExtractor()
    
    # Read image
    click.echo(f"Analyzing text in {actual_path}...")
    image = cv2.imread(actual_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        click.echo(f"Error: Could not read image {actual_path}")
        return
    
    # Segment into characters
    characters = segmenter.segment_characters(image)
    if not characters:
        click.echo("No characters detected in the image.")
        return
    
    click.echo(f"Detected {len(characters)} characters")
    
    # Extract character features
    char_features = []
    for char in characters:
        features = feature_extractor.extract_character_features(char['image'])
        char_features.append(features)
    
    # Make predictions
    X_char = np.vstack(char_features)
    
    if model.character_model:
        char_probas = model.character_model.predict_proba(X_char)
        char_predictions = []
        
        for i, proba in enumerate(char_probas):
            is_italic = proba[1] >= char_threshold
            char_predictions.append({
                'position': i,
                'is_italic': bool(is_italic),
                'confidence': float(proba[1])
            })
        
        # Count italic characters
        italic_count = sum(1 for pred in char_predictions if pred['is_italic'])
        
        click.echo(f"Result: {italic_count} of {len(characters)} characters are italic")
        click.echo(f"Character-level italic percentage: {italic_count / len(characters) * 100:.1f}%")
        
        # Determine overall word status - if over 50% of characters are italic, the word is italic
        word_is_italic = italic_count / len(characters) >= 0.5
        click.echo(f"Overall word classification: {'ITALIC' if word_is_italic else 'REGULAR'}")
        
        # Also use word-level model if available
        if model.word_level_model:
            # Extract word features
            word_features = feature_extractor.extract_word_features(image)
            word_probas = model.word_level_model.predict_proba(word_features.reshape(1, -1))[0]
            
            click.echo(f"Word-level model confidence: {word_probas[1]:.2f}")
            click.echo(f"Word-level classification: {'ITALIC' if word_probas[1] >= 0.6 else 'REGULAR'}")
    
    # Show visualization if requested
    if visualize:
        # Create visualization with character predictions
        vis_image = segmenter.visualize_segmentation(image, characters)
        
        # Add prediction labels
        for i, char in enumerate(characters):
            x, y, w, h = char['bbox']
            if char_predictions[i]['is_italic']:
                color = (0, 0, 255)  # Red for italic
                label = "I"
            else:
                color = (0, 255, 0)  # Green for regular
                label = "R"
            
            # Draw label
            cv2.putText(vis_image, label, (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw confidence
            conf_str = f"{char_predictions[i]['confidence']:.2f}"
            cv2.putText(vis_image, conf_str, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Save and show visualization
        vis_path = os.path.splitext(actual_path)[0] + "_char_detection.png"
        cv2.imwrite(vis_path, vis_image)
        click.echo(f"Visualization saved to {vis_path}")
        
        # Try to display the image
        try:
            from PIL import Image
            img = Image.open(vis_path)
            img.show()
        except Exception as e:
            click.echo(f"Could not display image: {e}")

@cli.command()
@click.argument('pdf_path_arg', required=False)
@click.option('--pdf-path', type=click.Path(exists=True), 
              help='Path to PDF to analyze')
@click.option('--output-path', type=click.Path(), default=None, 
              help='Path to save results JSON')
@click.option('--char-threshold', type=float, default=0.60,
              help='Confidence threshold for character-level classification')
def analyze_document_chars(pdf_path_arg, pdf_path, output_path, char_threshold):
    """Analyze a document with character-level detection.
    
    Can be called in two ways:
    
    italic-detector analyze-document-chars --pdf-path "document with spaces.pdf"
    
    OR simply:
    
    italic-detector analyze-document-chars "document with spaces.pdf"
    """
    from .data_prep import DataPreparation
    
    # Determine which path to use
    if pdf_path_arg and os.path.exists(pdf_path_arg):
        pdf_path = pdf_path_arg
    elif not pdf_path:
        click.echo("Error: Please provide a valid PDF path either as an argument or with --pdf-path")
        return
    
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + "_char_analysis.json"
    
    # Create temporary directory for text samples
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract text samples
        click.echo(f"Extracting text samples from {pdf_path}...")
        DataPreparation.extract_text_samples_from_pdf(
            pdf_path=pdf_path,
            output_dir=temp_dir
        )
        
        # Get all sample images
        sample_images = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
                       if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        click.echo(f"Analyzing {len(sample_images)} text samples...")
        
        # Initialize segmentation, feature extraction, and model
        segmenter = CharacterSegmentation()
        feature_extractor = CharacterFeatureExtractor()
        model = CharacterModel()
        
        # Load models
        try:
            model.load_models()
        except Exception as e:
            click.echo(f"Error loading models: {e}")
            click.echo("Please train the character model first with 'train-character-model'")
            return
        
        # Process each sample
        results = []
        for img_path in tqdm(sample_images):
            try:
                # Read image
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                # Segment into characters
                characters = segmenter.segment_characters(image)
                if not characters:
                    continue
                
                # Extract character features
                char_features = []
                for char in characters:
                    features = feature_extractor.extract_character_features(char['image'])
                    char_features.append(features)
                
                # Make character predictions
                if char_features:
                    X_char = np.vstack(char_features)
                    
                    if model.character_model:
                        char_probas = model.character_model.predict_proba(X_char)
                        char_predictions = []
                        
                        for i, proba in enumerate(char_probas):
                            is_italic = proba[1] >= char_threshold
                            char_predictions.append({
                                'position': i,
                                'is_italic': bool(is_italic),
                                'confidence': float(proba[1])
                            })
                        
                        # Count italic characters
                        italic_count = sum(1 for pred in char_predictions if pred['is_italic'])
                        
                        # Determine overall word status - if over 50% of characters are italic, the word is italic
                        word_is_italic = italic_count / len(characters) >= 0.5
                        char_confidence = italic_count / len(characters)
                        
                # Also use word-level model if available
                word_confidence = 0
                if model.word_level_model:
                    # Extract word features
                    word_features = feature_extractor.extract_word_features(image)
                    word_probas = model.word_level_model.predict_proba(word_features.reshape(1, -1))[0]
                    word_confidence = float(word_probas[1])
                
                # Combine character and word level predictions
                final_confidence = (char_confidence + word_confidence) / 2 if model.word_level_model else char_confidence
                final_is_italic = final_confidence >= 0.5
                
                # Store result
                sample_name = os.path.basename(img_path)
                results.append({
                    'sample': sample_name,
                    'is_italic': final_is_italic,
                    'confidence': final_confidence,
                    'char_level_italic_ratio': char_confidence,
                    'word_level_confidence': word_confidence if model.word_level_model else None,
                    'char_count': len(characters),
                    'italic_char_count': italic_count,
                    'path': img_path
                })
                
                # Create visualization for this sample
                vis_image = segmenter.visualize_segmentation(image, characters)
                
                # Add prediction labels
                for i, char in enumerate(characters):
                    x, y, w, h = char['bbox']
                    if char_predictions[i]['is_italic']:
                        color = (0, 0, 255)  # Red for italic
                        label = "I"
                    else:
                        color = (0, 255, 0)  # Green for regular
                        label = "R"
                    
                    # Draw label
                    cv2.putText(vis_image, label, (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Save visualization
                vis_dir = os.path.join(os.path.dirname(output_path), "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{os.path.splitext(sample_name)[0]}_char_detection.png")
                cv2.imwrite(vis_path, vis_image)
                
            except Exception as e:
                click.echo(f"Error processing {img_path}: {e}")
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Report summary
        italic_count = sum(1 for r in results if r.get('is_italic'))
        
        click.echo(f"\nAnalysis complete. Found {italic_count} italic text regions "
                 f"out of {len(results)} total.")
        click.echo(f"Results saved to {output_path}")
        if os.path.exists(vis_dir):
            click.echo(f"Visualizations saved to {vis_dir}")

def main():
    """Main entry point for the character-level detection application."""
    cli()

if __name__ == "__main__":
    main()