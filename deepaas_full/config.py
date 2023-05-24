"""Configuration loader for model source."""
import os
from pathlib import Path

MODELS_PATH = Path(os.getenv("MODELS_PATH", default="./models"))
DATA_PATH = Path(os.getenv("MODELS_PATH", default="./data"))
LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
INPUT_SHAPE = (IMAGE_SIZE**2,)
