"""Utilities module for API endpoints and methods.
"""
import logging
import os
import sys

from . import config

logger = logging.getLogger(__name__)


def ls_local(submodel: str):
    """Utility to return a list of models available in `models` folder.

    Arguments:
        submodel -- String with the submodel section in settings.

    Returns:
        A list of strings in the format {submodel}_{timestamp}.
    """
    logger.debug("Scanning at: %s/%s", config.MODELS_PATH, submodel)
    dirscan = os.scandir(f"{config.MODELS_PATH}/{submodel}")
    return [entry.name for entry in dirscan if entry.is_dir()]


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %d", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    get_args = generate_arguments(schema)
    def inject_function_schema(func):  # fmt: skip
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    get_args = generate_arguments(schema)
    def inject_function_schema(func):  # fmt: skip
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
