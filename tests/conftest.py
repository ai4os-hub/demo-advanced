"""Tests environment configuration."""
import os
import sys

# Configure environment variables to use models and datasets at tests
os.environ["DATA_PATH"] = "tests/datasets"
os.environ["MODELS_PATH"] = "tests/models"

# Makes vscode discover function capable of importing api from module
sys.path.insert(0, ".")
