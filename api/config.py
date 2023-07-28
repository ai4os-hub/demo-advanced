"""Configuration loader for DEEPaaS API."""
import os
from importlib.metadata import metadata as _metadata
from pathlib import Path


DATA_PATH = Path(os.getenv("DATA_PATH", default="./data/processed"))
MODEL_NAME = os.getenv("MODEL_NAME", default="deepaas_full")

MODEL_METADATA = _metadata(MODEL_NAME).json
