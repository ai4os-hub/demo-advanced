"""Configuration loader for model source."""
import os

LABEL_DIMENSIONS = int(os.getenv("LABEL_DIMENSIONS", default="10"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", default="28"))
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 1)
