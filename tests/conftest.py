"""Tests environment configuration."""
# pylint: disable=redefined-outer-name
import os
import shutil
import tempfile

import pytest

from api import config


@pytest.fixture(scope="session", autouse=True)
def check_mlflow_availability():
    """Fixture to skip tests if MLFlow is not available."""
    if not config.DEFAULT_MLFLOW_URI:
        pytest.skip("Undefined MLFLOW_TRACKING_URI env.")


@pytest.fixture(scope="session", autouse=True)
def original_datapath():
    """Fixture to generate a original directory path for datasets."""
    return config.DATA_PATH.absolute()


@pytest.fixture(scope="module", autouse=True, name="testdir")
def create_testdir(original_datapath):
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        shutil.copytree(original_datapath, f"{testdir}/{config.DATA_PATH}")
        yield os.chdir(testdir)
