"""Configuration loader for model source."""
import os

# Configuration of `models` path as default store for models
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", default=None)
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI or "models"

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
