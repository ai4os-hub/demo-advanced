"""Configuration loader for model source."""
import os

MODELS_PATH = os.getenv("MODELS_PATH", default="./models")
DATA_PATH = os.getenv("MODELS_PATH", default="./data")
LABEL_DIMENSIONS = os.getenv("MODELS_PATH", default="9")
