"""Italic text detector package."""

# Import main components
from .feature_extractor import FeatureExtractor
from .data_prep import DataPreparation
from .model import ItalicDetectionModel
from .ollama_integration import OllamaIntegration

# Import character-level components
from .character_segmentation import CharacterSegmentation
from .character_feature_extractor import CharacterFeatureExtractor
from .character_data_prep import CharacterDataPreparation
from .character_model import CharacterModel