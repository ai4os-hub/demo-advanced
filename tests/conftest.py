"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing. 
"""
# pylint: disable=redefined-outer-name
import os
import shutil
import tempfile

import pytest

# Project model configuration was imported inside the api as config
from api import config


@pytest.fixture(scope="session", autouse=True)
def original_datapath():
    """Fixture to generate a original directory path for datasets."""
    return config.DATA_PATH.absolute()


@pytest.fixture(scope="session", autouse=True)
def original_modelspath():
    """Fixture to generate a original directory path for datasets."""
    return config.MODELS_PATH.absolute()


@pytest.fixture(scope="module", name="testdir")
def create_testdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        os.chdir(testdir)
        yield testdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(testdir, original_datapath):
    """Fixture to copy the original data directory to the test directory."""
    shutil.copytree(original_datapath, f"{testdir}/{config.DATA_PATH}")


@pytest.fixture(scope="module", autouse=True)
def copytree_models(testdir, original_modelspath):
    """Fixture to copy the original models directory to the test directory."""
    shutil.copytree(original_modelspath, f"{testdir}/{config.MODELS_PATH}")
