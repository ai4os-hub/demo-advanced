"""Configuration loader for model source."""
import os
import random

MODELS_PATH = os.getenv("MODELS_PATH", default="./models")
DATA_PATH = os.getenv("MODELS_PATH", default="./data")
LABEL_DIMENSIONS = os.getenv("LABEL_DIMENSIONS", default="10")
RAND_SEED = os.getenv("RAND_SEED", f"{random.randint(1, 999999999)}")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", default="28")
