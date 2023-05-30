"""Tests environment configuration."""
import os
import pathlib
import shutil
import tempfile

import pytest

# Set tests models and data paths
models_path = pathlib.Path("tests/models")
models_path_origin = models_path.absolute()
data_path = pathlib.Path("tests/datasets")
data_path_origin = data_path.absolute()

# Configure environment variables to use models and datasets at tests
os.environ["MODELS_PATH"] = str(models_path)
os.environ["DATA_PATH"] = str(data_path)


@pytest.fixture(scope="module", autouse=True)
def _create_testdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        shutil.copytree(models_path_origin, f"{testdir}/{models_path}")
        shutil.copytree(data_path_origin, f"{testdir}/{data_path}")
        yield os.chdir(testdir)
