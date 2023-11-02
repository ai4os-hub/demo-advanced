"""Module to define CONSTANTS used across the AI-model package.

This module is used to define CONSTANTS used across the AI-model package.
Do not misuse this module to define variables that are not CONSTANTS or
exclusive to the demo_advanced package. You can use the `config.py`
inside `api` to define exclusive CONSTANTS related to your interface.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
# Do NOT import anything from `api` or `demo_advanced` packages here.
# That might create circular dependencies.
import os
from pathlib import Path


# DEEPaaS can load more than one installed models. Therefore, in order to
# avoid conflicts, each default PATH environment variables should lead to
# a different folder. The current practice is to use the path from where the
# model source is located.
BASE_PATH = Path(__file__).resolve(strict=True).parents[1]

# Path definition for the pre-trained models
MODELS_PATH = os.getenv("DEMO_ADVANCED_MODELS_PATH", BASE_PATH / "models")
MODELS_PATH = Path(MODELS_PATH)
# Path definition for data folder
DATA_PATH = os.getenv("DEMO_ADVANCED_DATA_PATH", BASE_PATH / "data")
DATA_PATH = Path(DATA_PATH)

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("DEMO_ADVANCED_LABEL_DIMENSIONS", "10"))
IMAGE_SIZE = int(os.getenv("DEMO_ADVANCED_IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
