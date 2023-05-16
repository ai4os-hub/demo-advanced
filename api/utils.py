"""Utilities module for API endpoints and methods.
"""
import logging
import os

from api import config

logger = logging.getLogger("__name__")


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
