"""Configuration loader for model source."""
import os
from importlib.metadata import metadata
from pathlib import Path

# Configuration of paths and model metadata
MODELS_PATH = Path(os.getenv("MODELS_PATH", default="models"))
DATA_PATH = Path(os.getenv("DATA_PATH", default="data"))
MODEL_NAME = os.getenv("MODEL_NAME", default="deepaas_full")
MODEL_METADATA = metadata(MODEL_NAME).json

# Configuration of `models` path as default store for models
_LOCAL_URI = str(Path.cwd() / MODELS_PATH)
_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", default=_LOCAL_URI)
os.environ["MLFLOW_TRACKING_URI"] = _MLFLOW_TRACKING_URI

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
