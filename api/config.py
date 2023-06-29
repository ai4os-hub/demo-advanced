"""Configuration loader for DEEPaaS API."""
import os
from importlib.metadata import metadata as _metadata
from pathlib import Path


DATA_PATH = Path(os.getenv("DATA_PATH", default="./data"))
MODEL_NAME = os.getenv("MODEL_NAME", default="deepaas_full")

MODEL_METADATA = _metadata(MODEL_NAME).json

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", default=None)
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID", default=None)
if MLFLOW_TRACKING_URI is None:  # Ignore EXPERIMENT_ID if MLFLOW_URI is not set
    MLFLOW_EXPERIMENT_ID = None
