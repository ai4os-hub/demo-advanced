"""Configuration loader for DEEPaaS API."""
import os
from importlib.metadata import metadata as _metadata

MODEL_NAME = os.getenv("MODEL_NAME", default="deepaas_full")
MODEL_METADATA = _metadata(MODEL_NAME).json
MODELS_PATH = os.getenv("MODELS_PATH", default="./models")
