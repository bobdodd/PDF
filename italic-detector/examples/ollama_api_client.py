#!/usr/bin/env python3
"""
Character-Level Italic Detection with Ollama REST API

This script demonstrates how to use the character-level italic detection model
deployed to Ollama via its REST API.
"""

import os
import sys
import json
import argparse
import requests
import time
import subprocess
import numpy as np
from typing import Dict, Any, List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.character_feature_extractor import CharacterFeatureExtractor
    from src.character_segmentation import CharacterSegmentation
except ImportError:
    print("Error: Could not import from src. Make sure you're running this script from the italic-detector directory.")
    sys.exit(1)

# Constants
DEFAULT_OLLAMA_API_URL = "http://localhost:11434/api"
MODEL_NAME = "char-italic-detector"
DEFAULT_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def check_ollama_running(api_url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Check if Ollama service is running."""
    try:
        response = requests.get(f"{api_url}/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_ollama(api_url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Start Ollama service if it's not running."""
    print("Attempting to start Ollama service...")
    try:
        # Start Ollama in the background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for service to be available
        for _ in range(30):  # wait up to 30 seconds
            time.sleep(1)
            if check_ollama_running(api_url):
                print("✓ Ollama service is now running")
                return True
        
        print("× Failed to start Ollama service within timeout")
        return False
    except Exception as e:
        print(f"× Error starting Ollama: {e}")
        return False

def check_model_available(model_name: str, api_url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Check if a model is available in Ollama."""
    try:
        response = requests.get(f"{api_url}/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"] == model_name for model in models)
        return False
    except requests.RequestException:
        return False

def start_model(model_name: str, api_url: str = DEFAULT_OLLAMA_API_URL) -> bool:
    """Pull the model if it's not available."""
    print(f"Pulling model {model_name}...")
    try:
        # Extract just the host:port part from the API URL
        from urllib.parse import urlparse
        parsed_url = urlparse(api_url)
        host_port = f"{parsed_url.netloc}"
        
        # Try to pull the model
        cmd = ["ollama", "pull", model_name]
        
        # Add --host flag if not using default localhost
        if host_port and host_port != "localhost:11434":
            cmd.extend(["--host", host_port])
            
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError:
        print(f"× Failed to pull model {model_name}")
        return False

def analyze_with_ollama_api(
    features: List[float], 
    model_name: str = MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT,
    api_url: str = DEFAULT_OLLAMA_API_URL
) -> Dict[str, Any]:
    """
    Analyze features using Ollama API.
    
    Args:
        features: List of feature values extracted from an image
        model_name: Name of the Ollama model to use
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary with analysis results
    """
    prompt = f"""
    Analyze these features extracted from a text image:
    {json.dumps(features)}
    
    Based on these features, is the text italic? Answer with 'Yes' if italic or 'No' if not italic.
    Also provide your confidence score between 0 and 1.
    Format your response as JSON: {{"is_italic": true/false, "confidence": 0.XX}}
    """
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 100
        }
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{api_url}/generate", 
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        return {
                            "error": "Could not parse JSON response",
                            "raw_response": response_text
                        }
                else:
                    # Fallback if JSON not found
                    is_italic = "yes" in response_text.lower()
                    return {
                        "is_italic": is_italic,
                        "confidence": 0.5,
                        "raw_response": response_text
                    }
            else:
                if attempt < MAX_RETRIES - 1:
                    print(f"Request failed with status {response.status_code}, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return {
                        "error": f"Request failed with status code: {response.status_code}",
                        "details": response.text
                    }
                    
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Request error: {e}, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
                continue
            else:
                return {"error": f"Request failed: {str(e)}"}
    
    return {"error": "All retry attempts failed"}

def analyze_image(
    image_path: str,
    model_name: str = MODEL_NAME,
    timeout: int = DEFAULT_TIMEOUT,
    visualize: bool = False,
    visualization_dir: str = '/tmp',
    api_url: str = DEFAULT_OLLAMA_API_URL
) -> Dict[str, Any]:
    """
    Analyze an image to detect if text is italic.
    
    Args:
        image_path: Path to the image file
        model_name: Name of the Ollama model to use
        timeout: Request timeout in seconds
        visualize: Whether to save visualization
        visualization_dir: Directory to save visualizations (default: /tmp)
        api_url: Ollama API URL (default: http://localhost:11434/api)
        
    Returns:
        Dictionary with analysis results
    """
    import os
    import cv2
    import numpy as np
    
    # Check if image exists
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}
        
    try:
        # Initialize components
        extractor = CharacterFeatureExtractor()
        segmenter = CharacterSegmentation()
        
        # Try to load the local model first for more accurate results
        try:
            from src.character_model import CharacterModel
            local_model = CharacterModel()
            local_model.load_models()
            print("Using local models for direct prediction")
            using_local_model = True
        except Exception as e:
            print(f"Could not load local models: {e}")
            print("Falling back to Ollama for prediction")
            using_local_model = False
        
        # Read image
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"error": f"Could not read image: {image_path}"}
            
        # Segment characters
        characters = segmenter.segment_characters(image)
        if not characters:
            return {"error": "No characters detected in the image"}
            
        print(f"Detected {len(characters)} characters")
        
        # Extract character features
        char_features = []
        for char in characters:
            features = extractor.extract_character_features(char['image'])
            char_features.append(features)
            
        # Extract word features
        word_features = extractor.extract_word_features(image)
        
        # Process with local models or Ollama
        results = []
        word_result = {}
        
        if using_local_model:
            # Process with local models
            X_char = np.vstack(char_features)
            char_probas = local_model.character_model.predict_proba(X_char)
            
            for i, proba in enumerate(char_probas):
                is_italic = proba[1] >= 0.75  # Higher threshold for better precision
                results.append({
                    "character_index": i,
                    "is_italic": bool(is_italic),
                    "confidence": float(proba[1])
                })
            
            # Word-level prediction if available
            if local_model.word_level_model is not None:
                word_probas = local_model.word_level_model.predict_proba(word_features.reshape(1, -1))[0]
                word_result = {
                    "is_italic": bool(word_probas[1] >= 0.6),
                    "confidence": float(word_probas[1])
                }
            else:
                # Fall back to aggregating character results
                italic_ratio = sum(1 for r in results if r["is_italic"]) / len(results) if results else 0
                word_result = {
                    "is_italic": italic_ratio >= 0.5,
                    "confidence": max(0.5, italic_ratio)
                }
        else:
            # Process with Ollama
            for i, features in enumerate(char_features):
                character_result = analyze_with_ollama_api(
                    features.tolist(),
                    model_name=model_name,
                    timeout=timeout,
                    api_url=api_url
                )
                results.append({
                    "character_index": i,
                    **character_result
                })
                
            # Analyze whole word with Ollama
            word_result = analyze_with_ollama_api(
                word_features.tolist(),
                model_name=model_name,
                timeout=timeout,
                api_url=api_url
            )
        
        # Count italic characters
        italic_chars = [result for result in results if result.get("is_italic", False)]
        italic_percentage = len(italic_chars) / len(characters) * 100 if characters else 0
        
        # Override word level result based on character detection evidence
        # Case 1: Strong character evidence for italic text (>80%)
        if italic_percentage > 80:
            # Override word-level result when character detection strongly suggests italic
            if not word_result.get("is_italic", False):
                word_result["is_italic"] = True
                word_result["confidence"] = max(word_result.get("confidence", 0.5), 0.6)
                word_result["note"] = "Overridden by character-level detection (strong italic evidence)"
        
        # Case 2: Word-level model has very high confidence for italic (>0.8)
        # Trust the word-level model when it's very confident about italic
        elif word_result.get("confidence", 0) > 0.8 and word_result.get("is_italic", False):
            # Just ensure the is_italic flag is set to True
            word_result["is_italic"] = True
            word_result["note"] = "Using high-confidence word-level prediction"
            
        # Case 3: Strong character evidence for regular text (<=40% italic)
        # Only apply if word-level model is not highly confident about italic
        elif italic_percentage <= 40 and word_result.get("confidence", 0) <= 0.8:
            # Override word-level result when character detection strongly suggests regular
            if word_result.get("is_italic", False):
                word_result["is_italic"] = False
                word_result["confidence"] = max(1.0 - word_result.get("confidence", 0.5), 0.6)
                word_result["note"] = "Overridden by character-level detection (strong regular evidence)"
        
        # Case 4: Moderate italic evidence (>50% italic) with low word-level confidence
        elif italic_percentage > 50 and word_result.get("confidence", 0.5) < 0.5:
            # Override if word level has low confidence
            word_result["is_italic"] = True
            word_result["confidence"] = max(word_result.get("confidence", 0), 0.55)
            word_result["note"] = "Overridden by character-level detection (majority of characters italic)"
        
        # Create summary
        summary = {
            "word_result": word_result,
            "character_results": results,
            "total_characters": len(characters),
            "italic_characters": len(italic_chars),
            "italic_percentage": italic_percentage,
            "overall_italic": word_result.get("is_italic", False)  # Use word-level result as overall (now potentially overridden)
        }
        
        # Create visualization if requested
        if visualize:
            # Create visualization with character predictions
            vis_image = segmenter.visualize_segmentation(image, characters)
            
            # Add prediction labels
            for i, char in enumerate(characters):
                x, y, w, h = char['bbox']
                result = results[i]
                
                if result.get("is_italic", False):
                    color = (0, 0, 255)  # Red for italic
                    label = "I"
                else:
                    color = (0, 255, 0)  # Green for regular
                    label = "R"
                
                # Draw label
                cv2.putText(vis_image, label, (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw confidence
                conf_str = f"{result.get('confidence', 0):.2f}"
                cv2.putText(vis_image, conf_str, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Add overall result to image
            overall_text = f"Overall: {'ITALIC' if summary['overall_italic'] else 'REGULAR'}"
            cv2.putText(vis_image, overall_text, (10, image.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 0, 255) if summary['overall_italic'] else (0, 255, 0), 1)
            
            # Ensure visualization directory exists
            import os
            os.makedirs(visualization_dir, exist_ok=True)
                
            # Save visualization
            vis_filename = os.path.basename(image_path)
            vis_basename, vis_ext = os.path.splitext(vis_filename)
            vis_path = os.path.join(visualization_dir, f"{vis_basename}_detection{vis_ext}")
            
            # Ensure the image is saved in the right format
            cv2.imwrite(vis_path, vis_image)
            
            # Also try to display the image
            try:
                from PIL import Image
                img = Image.open(vis_path)
                img.show()
            except Exception as e:
                print(f"Could not display image: {e}")
                
            summary["visualization_path"] = vis_path
            
        return summary
        
    except Exception as e:
        import traceback
        return {
            "error": f"Error analyzing image: {str(e)}",
            "traceback": traceback.format_exc()
        }

def main():
    """Main function to demonstrate Ollama API usage."""
    import os
    import cv2
    import numpy as np
    
    parser = argparse.ArgumentParser(description='Character-Level Italic Detection with Ollama')
    parser.add_argument('image_path', help='Path to the image to analyze')
    parser.add_argument('--model', default=MODEL_NAME, help=f'Ollama model name (default: {MODEL_NAME})')
    parser.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT, help=f'API timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization of the detection')
    parser.add_argument('--visualization-dir', default='/tmp', help='Directory to save visualizations (default: /tmp)')
    parser.add_argument('--use-local', action='store_true', help='Force use of local models (skip Ollama)')
    parser.add_argument('--api-url', default=DEFAULT_OLLAMA_API_URL, help=f'Ollama API URL (default: {DEFAULT_OLLAMA_API_URL})')
    args = parser.parse_args()
    
    # Skip Ollama checks if using local models
    if not args.use_local:
        # Check if Ollama is running
        if not check_ollama_running(args.api_url):
            print("× Ollama service is not running")
            if not start_ollama(args.api_url):
                print("Could not start Ollama service. Please ensure Ollama is installed.")
                print("Using local models instead...")
                args.use_local = True
        else:
            print("✓ Ollama service is running")
        
        # Check if the model is available
        if not args.use_local and not check_model_available(args.model, args.api_url):
            print(f"× Model '{args.model}' is not available")
            if not start_model(args.model, args.api_url):
                print(f"Could not pull model '{args.model}'. Make sure you've deployed it with 'char-detector deploy-to-ollama'")
                print("Using local models instead...")
                args.use_local = True
        elif not args.use_local:
            print(f"✓ Model '{args.model}' is available")
    else:
        print("Using local models as requested (skipping Ollama)")
    
    # Analyze the image
    print(f"Analyzing image: {args.image_path}")
    
    # Load local models directly if requested
    if args.use_local:
        try:
            import cv2
            import numpy as np
            import os
            from src.character_model import CharacterModel
            model = CharacterModel()
            model.load_models()
            
            from src.character_segmentation import CharacterSegmentation
            from src.character_feature_extractor import CharacterFeatureExtractor
            
            segmenter = CharacterSegmentation()
            extractor = CharacterFeatureExtractor()
            
            # Read and analyze image
            image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Error: Could not read image {args.image_path}")
                return 1
                
            # Segment characters
            characters = segmenter.segment_characters(image)
            if not characters:
                print("No characters detected in the image")
                return 1
                
            print(f"Detected {len(characters)} characters")
            
            # Extract features and classify
            char_features = []
            for char in characters:
                features = extractor.extract_character_features(char['image'])
                char_features.append(features)
                
            # Make character predictions
            X_char = np.vstack(char_features)
            char_probas = model.character_model.predict_proba(X_char)
            
            # Count italic characters
            char_predictions = []
            for i, proba in enumerate(char_probas):
                is_italic = proba[1] >= 0.75
                char_predictions.append({
                    'position': i,
                    'is_italic': bool(is_italic),
                    'confidence': float(proba[1])
                })
                
            italic_count = sum(1 for pred in char_predictions if pred['is_italic'])
            print(f"Result: {italic_count} of {len(characters)} characters are italic")
            print(f"Character-level italic percentage: {italic_count / len(characters) * 100:.1f}%")
            
            # Word-level prediction
            word_is_italic = False
            if model.word_level_model:
                word_features = extractor.extract_word_features(image)
                word_probas = model.word_level_model.predict_proba(word_features.reshape(1, -1))[0]
                
                # Calculate the percentage of characters classified as italic
                italic_percentage = italic_count / len(characters) * 100 if characters else 0
                
                # Default word-level prediction
                word_is_italic = word_probas[1] >= 0.6
                override_reason = None
                
                # Case 1: Strong character evidence for italic text (>80%)
                if italic_percentage > 80:
                    if not word_is_italic:  # Only override if there's a disagreement
                        word_is_italic = True
                        override_reason = "strong italic evidence"
                
                # Case 2: Word-level model has very high confidence for italic (>0.8)
                elif word_probas[1] > 0.8:
                    # Trust the word-level model when it's very confident
                    word_is_italic = True
                    override_reason = "high-confidence word-level prediction"
                
                # Case 3: Strong character evidence for regular text (<=40%)
                # Only apply if word-level model is not highly confident about italic
                elif italic_percentage <= 40 and word_probas[1] <= 0.8:
                    if word_is_italic:  # Only override if there's a disagreement
                        word_is_italic = False
                        override_reason = "strong regular evidence"
                
                # Case 4: Moderate italic evidence (>50% italic) with low word-level confidence
                elif italic_percentage > 50 and word_probas[1] < 0.5:
                    if not word_is_italic:  # Only override if there's a disagreement
                        word_is_italic = True
                        override_reason = "majority of characters italic"
                
                # Print appropriate message
                if override_reason:
                    print(f"Word-level model confidence: {word_probas[1]:.2f} (overridden by character detection: {override_reason})")
                else:
                    print(f"Word-level model confidence: {word_probas[1]:.2f}")
                
                print(f"Word-level classification: {'ITALIC' if word_is_italic else 'REGULAR'}")
                
            # Overall classification (prefer character-level when very confident)
            overall_is_italic = word_is_italic
            print(f"Overall classification: {'ITALIC' if overall_is_italic else 'REGULAR'}")
            
            # Create visualization if requested
            if args.visualize:
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
                # Ensure visualization directory exists
                os.makedirs(args.visualization_dir, exist_ok=True)
                
                vis_filename = os.path.basename(args.image_path)
                vis_basename, vis_ext = os.path.splitext(vis_filename)
                vis_path = os.path.join(args.visualization_dir, 
                                       f"{vis_basename}_detection{vis_ext}")
                cv2.imwrite(vis_path, vis_image)
                print(f"Visualization saved to {vis_path}")
                
                # Try to display the image
                try:
                    from PIL import Image
                    img = Image.open(vis_path)
                    img.show()
                except Exception as e:
                    print(f"Could not display image: {e}")
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
            return 1
            
        return 0
        
    # Use Ollama API
    result = analyze_image(
        args.image_path,
        model_name=args.model,
        timeout=args.timeout,
        visualize=args.visualize,
        visualization_dir=args.visualization_dir,
        api_url=args.api_url
    )
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
        if "traceback" in result:
            print("\nTraceback:")
            print(result["traceback"])
    else:
        print("\nResults:")
        print(f"Total characters: {result['total_characters']}")
        print(f"Italic characters: {result['italic_characters']} ({result['italic_percentage']:.1f}%)")
        result_note = f" - {result['word_result'].get('note', '')}" if 'note' in result['word_result'] else ""
        print(f"Word-level result: {'ITALIC' if result['word_result'].get('is_italic', False) else 'REGULAR'} "
              f"(confidence: {result['word_result'].get('confidence', 0):.2f}){result_note}")
        print(f"Overall classification: {'ITALIC' if result['overall_italic'] else 'REGULAR'}")
        
        if "visualization_path" in result:
            print(f"\nVisualization saved to: {result['visualization_path']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())