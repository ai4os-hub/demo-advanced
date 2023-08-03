"""Utilities module for API endpoints and methods.
"""
import logging
import sys
from pathlib import Path

import mlflow

from deepaas_full import config

logger = logging.getLogger(__name__)
mlflow_client = mlflow.MlflowClient(config.MODELS_PATH)


def ls_models():
    """Utility to return a list of models available in the MLFlow connected
    models registry.

    Returns:
        A list of RegisteredModel from mlflow.
    """
    models = mlflow_client.search_registered_models()
    return {x.name: x.description for x in models}


def ls_datasets():
    """Utility to return a list of datasets available in `data` folder.

    Returns:
        A list of strings in the format {id}-{type}.npz.
    """
    processed_path = Path(config.DATA_PATH) / "processed"
    logger.debug("Scanning at: %s", processed_path)
    dirscan = (x.name for x in processed_path.glob("*.npz"))
    return sorted(dirscan)


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
