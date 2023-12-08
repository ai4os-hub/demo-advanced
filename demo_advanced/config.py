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
import mlflow


# DEEPaaS can load more than one installed models. Therefore, in order to
# avoid conflicts, each default PATH environment variables should lead to
# a different folder. The current practice is to use the path from where the
# model source is located.

# Path definition for the pre-trained models
MODELS_URI = os.getenv("DEMO_ADVANCED_MODELS_URI", "models")
DATA_URI = os.getenv("DEMO_ADVANCED_DATA_URI", "data")

# MLFlow configuration
mlflow.tensorflow.autolog()
mlflow_client = mlflow.tracking.MlflowClient()

# Path definition for data folder

# Configuration of model framework features
LABEL_DIMENSIONS = int(os.getenv("DEMO_ADVANCED_LABEL_DIMENSIONS", "10"))
IMAGE_SIZE = int(os.getenv("DEMO_ADVANCED_IMAGE_SIZE", default="28"))
IMAGES_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
