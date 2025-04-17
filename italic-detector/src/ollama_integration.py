import os
import json
import subprocess
import numpy as np
from typing import Dict, Any, List

class OllamaIntegration:
    """Integration with Ollama for model deployment."""
    
    def __init__(self, 
                 model_name: str = "italic-detector",
                 base_model: str = "llama2",
                 onnx_model_path: str = "models/saved/italic_detector.onnx"):
        """Initialize Ollama integration."""
        self.model_name = model_name
        self.base_model = base_model
        self.onnx_model_path = onnx_model_path
    
    def _check_ollama_installed(self) -> bool:
        """Check if Ollama is installed."""
        try:
            # Try with full path search
            import os
            import shutil
            
            # Try to find ollama in the PATH
            ollama_path = shutil.which("ollama")
            
            if ollama_path:
                # Use the full path if found
                result = subprocess.run([ollama_path, "list"], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE, 
                                      check=False)
                if result.returncode == 0:
                    return True
            
            # Try direct command as fallback
            result = subprocess.run(["ollama", "list"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  check=False)
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error checking Ollama: {e}")
            return False
    
    def create_modelfile(self, modelfile_path: str = "Modelfile") -> str:
        """Create a Modelfile for Ollama."""
        print(f"Creating Modelfile at {modelfile_path}...")
        
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model file not found: {self.onnx_model_path}")
        
        # Create the system prompt with proper escaping
        system_prompt = (
            "You are an OCR assistant specializing in detecting italic text in images.\n"
            "You analyze features such as stroke angle, text slant, and character shape to determine if text is italic.\n"
            "When given extracted features from a text image, you will predict if the text is italic or not.\n"
            'When using the model in the future, respond with JSON in the format: {"is_italic": true/false, "confidence": 0.XX}'
        )
        
        # Write the Modelfile
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM {self.base_model}\n\n")
            f.write("# System prompt for text style detection\n")
            f.write(f'SYSTEM """{system_prompt}"""\n\n')
            f.write("# Parameters\n")
            f.write("PARAMETER temperature 0.1\n")
        
        print(f"Modelfile created at {modelfile_path}")
        return modelfile_path
    
    def build_model(self, modelfile_path: str = "Modelfile") -> bool:
        """Build the Ollama model."""
        # Check if Ollama is installed
        if not self._check_ollama_installed():
            import shutil
            ollama_path = shutil.which("ollama")
            raise RuntimeError(
                f"Ollama not found or not working. Path search result: {ollama_path}. "
                f"Please ensure Ollama is installed and running (https://ollama.ai). "
                f"Try running 'ollama list' in your terminal to check if it's working."
            )
        
        print(f"Building Ollama model: {self.model_name}...")
        
        # Create modelfile if it doesn't exist
        if not os.path.exists(modelfile_path):
            modelfile_path = self.create_modelfile(modelfile_path)
        
        # Find ollama path
        import shutil
        ollama_path = shutil.which("ollama") or "ollama"
        
        # Build the model
        try:
            print(f"Running: {ollama_path} create {self.model_name} -f {modelfile_path}")
            result = subprocess.run(
                [ollama_path, "create", self.model_name, "-f", modelfile_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print(result.stdout)
            print(f"Successfully built Ollama model: {self.model_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error building Ollama model: {e}")
            print(f"Stderr: {e.stderr}")
            return False
    
    def run_prediction(self, features: List[float]) -> Dict[str, Any]:
        """Run a prediction using the Ollama model."""
        # Check if Ollama is installed
        if not self._check_ollama_installed():
            import shutil
            ollama_path = shutil.which("ollama")
            raise RuntimeError(
                f"Ollama not found or not working. Path search result: {ollama_path}. "
                f"Please ensure Ollama is installed and running (https://ollama.ai). "
                f"Try running 'ollama list' in your terminal to check if it's working."
            )
        
        # Fall back to local model prediction if ONNX model exists
        try:
            import onnxruntime as ort
            from sklearn.ensemble import RandomForestClassifier
            import pickle
            
            # First check for pickle model
            model_path = "models/saved/italic_detector.pkl"
            if os.path.exists(model_path):
                print(f"Using local model for prediction ({model_path})")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                # Convert features to numpy array
                features_array = np.array(features).reshape(1, -1)
                
                # Get probability of italic
                proba = model.predict_proba(features_array)[0]
                
                # Balanced threshold for classifying as italic
                is_italic = proba[1] >= 0.54
                
                return {
                    "is_italic": bool(is_italic), 
                    "confidence": float(proba[1])
                }
            
            # Then check for ONNX model
            onnx_path = "models/saved/italic_detector.onnx"
            if os.path.exists(onnx_path):
                print(f"Using ONNX model for prediction ({onnx_path})")
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                
                # Prepare input data
                input_data = {input_name: np.array([features], dtype=np.float32)}
                
                # Run inference
                pred_onnx = session.run(None, input_data)
                is_italic = bool(pred_onnx[0][0])
                confidence = float(0.8 if is_italic else 0.2)
                
                return {
                    "is_italic": is_italic,
                    "confidence": confidence
                }
                
        except ImportError:
            # If onnxruntime is not available, fall back to Ollama
            pass
            
        # Prepare input for Ollama
        prompt = f"""
Analyze these features extracted from a text image:
{json.dumps(features)}

Based on these features, is the text italic? Answer with 'Yes' if italic or 'No' if not italic.
Also provide your confidence score between 0 and 1.
Use balanced judgment to determine if the text is italic or not.
Format your response as JSON: {{"is_italic": true/false, "confidence": 0.XX}}
"""
        
        # Find ollama path
        import shutil
        ollama_path = shutil.which("ollama") or "ollama"
        
        # Run Ollama prediction
        try:
            print(f"Running prediction with: {ollama_path} run {self.model_name}")
            result = subprocess.run(
                [ollama_path, "run", self.model_name, prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Extract JSON from response
            response = result.stdout
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                prediction = json.loads(json_str)
            else:
                # Fallback if JSON not found
                is_italic = "yes" in response.lower()
                prediction = {
                    "is_italic": is_italic,
                    "confidence": 0.5,
                    "raw_response": response
                }
            
            return prediction
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Ollama prediction: {e}")
            return {
                "error": str(e)
            }