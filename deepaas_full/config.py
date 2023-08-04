"""Configuration loader for model source."""
import os

# Configuration of paths and model metadata
MODELS_PATH = os.getenv("MODELS_PATH", default="models")
DATA_PATH = os.getenv("DATA_PATH", default="data")

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
