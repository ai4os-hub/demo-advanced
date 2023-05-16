"""Configuration loader for DEEPaaS API."""
import configparser
import os
import pathlib
from importlib.metadata import metadata as _metadata

# Get configuration from user env and merge with pkg settings
# TODO: Repalace YOURMODELAPI_SETTINGS by other variable name
SETTINGS_FILE = pathlib.Path(__file__).parent / "settings.ini"
SETTINGS_FILE = os.getenv("YOURMODELAPI_SETTINGS", default=SETTINGS_FILE)
settings = configparser.ConfigParser()
settings.read(SETTINGS_FILE)

try:  # Configure model metadata from pkg metadata and backbones
    MODEL_NAME = os.getenv("MODEL_NAME", default=settings["model"]["name"])
    MODEL_METADATA = _metadata(MODEL_NAME).json
except KeyError as err:
    raise RuntimeError("Undefined configuration for [model]name") from err

try:  # Configure models folder for prediction and training arguments
    MODELS_PATH = os.getenv("MODELS_PATH", default=settings["models"]["path"])
    os.environ["MODELS_PATH"] = MODELS_PATH
except KeyError as err:
    raise RuntimeError("Undefined configuration for [models]path") from err
