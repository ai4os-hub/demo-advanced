"""Configuration loader for model source."""
import os
from importlib import metadata

# Configuration of paths and model metadata
MODELS_PATH = os.getenv("MODELS_PATH", default="models")
DATA_PATH = os.getenv("DATA_PATH", default="data")
MODEL_NAME = os.getenv("MODEL_NAME", default="deepaas_full")
MODEL_METADATA = metadata.metadata(MODEL_NAME).json

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
