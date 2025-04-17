import os
import sys
import click
import pandas as pd
import numpy as np
import tempfile
from tqdm import tqdm

# Fix imports to work both as package and as script
try:
    from .feature_extractor import FeatureExtractor
    from .data_prep import DataPreparation
    from .model import ItalicDetectionModel
    from .ollama_integration import OllamaIntegration
    from .data_collector import collect_labels
except ImportError:
    from feature_extractor import FeatureExtractor
    from data_prep import DataPreparation
    from model import ItalicDetectionModel
    from ollama_integration import OllamaIntegration
    from data_collector import collect_labels

@click.group()
def cli():
    """Italic Text Detector - Detect italic text in scanned documents."""
    pass

@cli.command()
@click.option('--pdf-folder', type=click.Path(exists=True), required=True, 
              help='Path to folder containing PDF files to process')
@click.option('--temp-dir', type=click.Path(), default=None, 
              help='Directory to store temporary files (default: system temp directory)')
@click.option('--output-dir', type=click.Path(), default='data/extracted', 
              help='Directory to save extracted samples')
@click.option('--min-size', type=int, default=15, 
              help='Minimum height for text regions')
def extract_pdf_data(pdf_folder, temp_dir, output_dir, min_size):
    """Extract text samples from a folder of PDF files."""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="italic_detector_")
        click.echo(f"Created temporary directory: {temp_dir}")
    
    click.echo(f"Processing PDFs from {pdf_folder}")
    click.echo(f"Temporary files will be stored in {temp_dir}")
    click.echo(f"Extracted samples will be saved to {output_dir}")
    
    try:
        total_samples = DataPreparation.process_pdf_folder(
            pdf_folder=pdf_folder,
            output_dir=output_dir,
            temp_dir=temp_dir,
            min_size=min_size
        )
        
        click.echo(f"Processing complete! Extracted {total_samples} text samples.")
        click.echo(f"Samples are ready for labeling in {output_dir}")
    except Exception as e:
        click.echo(f"Error processing PDFs: {e}")

@cli.command()
@click.option('--source-dir', type=click.Path(exists=True), required=True, 
              help='Directory containing text samples to label')
@click.option('--italic-dir', type=click.Path(), default='data/italic', 
              help='Directory to save italic samples')
@click.option('--regular-dir', type=click.Path(), default='data/spare_regular', 
              help='Directory to save regular samples')
@click.option('--temp-dir', type=click.Path(), default=None, 
              help='Directory to store temporary files (default: system temp directory)')
def label_samples(source_dir, italic_dir, regular_dir, temp_dir):
    """Label text samples as italic or regular."""
    click.echo(f"Starting labeling session for samples in {source_dir}")
    click.echo("You will be shown each image and asked to classify it.")
    click.echo("Press '1' for italic, '2' for regular, any other key to skip.")
    
    try:
        collect_labels(
            source_dir=source_dir,
            italic_dir=italic_dir,
            regular_dir=regular_dir,
            temp_dir=temp_dir
        )
        
        click.echo("Labeling session complete!")
        click.echo(f"Italic samples saved to: {italic_dir}")
        click.echo(f"Regular samples saved to: {regular_dir}")
    except Exception as e:
        click.echo(f"Error during labeling: {e}")

@cli.command()
@click.argument('pdf_path_arg', required=False)
@click.option('--pdf-path', type=click.Path(exists=True), help='Path to PDF to extract samples from')
@click.option('--output-dir', type=click.Path(), default=None, help='Directory to save samples')
def extract_samples(pdf_path_arg, pdf_path, output_dir):
    """Extract text samples from a PDF for dataset creation.
    
    Can be called in two ways:
    
    italic-detector extract-samples --pdf-path "document with spaces.pdf"
    
    OR simply:
    
    italic-detector extract-samples "document with spaces.pdf"
    """
    # Determine which path to use
    if pdf_path_arg and os.path.exists(pdf_path_arg):
        pdf_path = pdf_path_arg
    elif not pdf_path:
        click.echo("Error: Please provide a valid PDF path either as an argument or with --pdf-path")
        return
        
    if output_dir is None:
        output_dir = "data/test_samples"
    
    DataPreparation.extract_text_samples_from_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir
    )

@cli.command()
@click.option('--test-size', type=float, default=0.2, help='Proportion of data for testing')
@click.option('--augment', is_flag=True, help='Apply data augmentation to increase dataset size')
def prepare_data(test_size, augment):
    """Prepare dataset for training."""
    data_prep = DataPreparation()
    
    if augment:
        click.echo("Data augmentation enabled - will create multiple variations of each sample")
        click.echo("This will significantly increase processing time but produce a larger dataset")
    
    dataset = data_prep.create_dataset(test_size=test_size, augment=augment)
    
    click.echo(f"Dataset prepared with {len(dataset['train'])} training samples and {len(dataset['test'])} test samples")

@cli.command()
def train_model():
    """Train the italic detection model."""
    # Check if processed data exists
    train_path = "data/processed/train_data.csv"
    test_path = "data/processed/test_data.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        click.echo("Processed data not found. Run 'prepare-data' first.")
        return
    
    # Load data
    click.echo("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Train model
    model = ItalicDetectionModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model_file = model.save()
    click.echo(f"Model saved to {model_file}")
    
    # Export to ONNX
    onnx_file = model.export_to_onnx()
    click.echo(f"Model exported to ONNX format: {onnx_file}")

@cli.command()
@click.option('--model-name', default='italic-detector', help='Name for the Ollama model')
@click.option('--base-model', default='llama2', help='Base LLM to use')
@click.option('--onnx-model', type=click.Path(exists=True), default=None, help='Path to ONNX model file')
def deploy_to_ollama(model_name, base_model, onnx_model):
    """Deploy the model to Ollama."""
    if onnx_model is None:
        onnx_model = "models/saved/italic_detector.onnx"
        if not os.path.exists(onnx_model):
            click.echo("ONNX model not found. Train the model first.")
            return
    
    click.echo(f"Deploying model {onnx_model} to Ollama as {model_name}...")
    
    # Create Ollama integration
    ollama = OllamaIntegration(
        model_name=model_name,
        base_model=base_model,
        onnx_model_path=onnx_model
    )
    
    # Build the model
    success = ollama.build_model()
    
    if success:
        click.echo(f"Model successfully deployed to Ollama as {model_name}")
    else:
        click.echo("Model deployment failed.")

@cli.command()
@click.argument('image_path_arg', required=False)
@click.option('--image-path', type=click.Path(exists=True), 
              help='Path to text image to analyze (can be in double quotes without escaping spaces)')
@click.option('--model-name', default='italic-detector', help='Name of the Ollama model to use')
def detect_italic(image_path_arg, image_path, model_name):
    """Detect if text in an image is italic.
    
    Can be called in two ways:
    
    italic-detector detect-italic --image-path "path with spaces.png"
    
    OR simply:
    
    italic-detector detect-italic "path with spaces.png"
    """
    # Determine which path to use
    if image_path_arg and os.path.exists(image_path_arg):
        actual_path = image_path_arg
    elif image_path:
        actual_path = image_path
    else:
        click.echo("Error: Please provide a valid image path either as an argument or with --image-path")
        return
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features_from_file(actual_path)
    
    # Show which path we're using
    click.echo(f"Analyzing text in {actual_path}...")
    
    # Create Ollama integration
    ollama = OllamaIntegration(model_name=model_name)
    
    # Run prediction
    result = ollama.run_prediction(features.tolist())
    
    # Display result
    if "error" in result:
        click.echo(f"Error: {result['error']}")
    else:
        is_italic = result.get('is_italic', False)
        confidence = result.get('confidence', 0)
        
        click.echo(f"Result: {'ITALIC' if is_italic else 'REGULAR'} text")
        click.echo(f"Confidence: {confidence:.2f}")

@cli.command()
@click.argument('pdf_path_arg', required=False)
@click.option('--pdf-path', type=click.Path(exists=True), help='Path to PDF to analyze')
@click.option('--output-path', type=click.Path(), default=None, help='Path to save results JSON')
@click.option('--model-name', default='italic-detector', help='Name of the Ollama model to use')
def process_document(pdf_path_arg, pdf_path, output_path, model_name):
    """Process a document to detect italic text.
    
    Can be called in two ways:
    
    italic-detector process-document --pdf-path "document with spaces.pdf"
    
    OR simply:
    
    italic-detector process-document "document with spaces.pdf"
    """
    # Determine which path to use
    if pdf_path_arg and os.path.exists(pdf_path_arg):
        pdf_path = pdf_path_arg
    elif not pdf_path:
        click.echo("Error: Please provide a valid PDF path either as an argument or with --pdf-path")
        return
    import json
    import tempfile
    
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + "_italic_analysis.json"
    
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
        
        # Extract features
        extractor = FeatureExtractor()
        
        # Create Ollama integration
        ollama = OllamaIntegration(model_name=model_name)
        
        # Process each sample
        results = []
        for img_path in tqdm(sample_images):
            try:
                # Extract features
                features = extractor.extract_features_from_file(img_path)
                
                # Run prediction
                prediction = ollama.run_prediction(features.tolist())
                
                # Store result
                sample_name = os.path.basename(img_path)
                results.append({
                    'sample': sample_name,
                    'is_italic': prediction.get('is_italic', False),
                    'confidence': prediction.get('confidence', 0),
                    'path': img_path
                })
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

def main():
    """Main entry point for the application."""
    cli()

if __name__ == "__main__":
    main()