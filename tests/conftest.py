"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing. 
"""
# pylint: disable=redefined-outer-name
import inspect
import os
import shutil
import tempfile

import pytest

import api


@pytest.fixture(scope="session", autouse=True)
def original_datapath():
    """Fixture to generate a original directory path for datasets."""
    return api.config.DATA_PATH.absolute()


@pytest.fixture(scope="session", autouse=True)
def original_modelspath():
    """Fixture to generate a original directory path for datasets."""
    return api.config.MODELS_PATH.absolute()


@pytest.fixture(scope="module", name="testdir")
def create_testdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        os.chdir(testdir)
        yield testdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(testdir, original_datapath):
    """Fixture to copy the original data directory to the test directory."""
    shutil.copytree(original_datapath, f"{testdir}/{api.config.DATA_PATH}")


@pytest.fixture(scope="module", autouse=True)
def copytree_models(testdir, original_modelspath):
    """Fixture to copy the original models directory to the test directory."""
    shutil.copytree(original_modelspath, f"{testdir}/{api.config.MODELS_PATH}")


def generate_signature(names, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD):
    """Function to generate dynamically signatures."""
    parameters = [inspect.Parameter(name, kind) for name in names]
    return inspect.Signature(parameters=parameters)


def generate_fields_fixture(signature):
    """Function to generate dynamically fixtures."""
    def fixture_function(**options):  # fmt: skip
        return {k: v for k, v in options.items() if v is not None}
    fixture_function.__signature__ = signature
    return pytest.fixture(scope="module")(fixture_function)


# Generate and inject fixtures for predict arguments
fields_predict = api.schemas.PredArgsSchema().fields
signature = generate_signature(fields_predict.keys())
globals()["predict_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return defined api metadata."""
    return api.get_metadata()


@pytest.fixture(scope="module")
def predictions(predict_kwds):
    """Fixture to return predictions to assert properties."""
    return api.predict(**predict_kwds)
