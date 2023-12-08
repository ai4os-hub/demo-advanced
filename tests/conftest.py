"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing.

Modify this file only if you need to add new fixtures or modify the existing
related to the environment and generic tests.
"""
# pylint: disable=redefined-outer-name
import inspect
import os
import pathlib
import shutil
import tempfile

import pytest

import api


@pytest.fixture(scope="session", autouse=True)
def path_testdata():
    """Fixture to generate absolute path for <repo>/test/datas."""
    return pathlib.Path(api.config.DATA_URI).absolute()


@pytest.fixture(scope="session", autouse=True)
def path_testmodels():
    """Fixture to generate absolute path for <repo>/test/models."""
    return pathlib.Path(api.config.MODELS_URI).absolute()


@pytest.fixture(scope="session", autouse=True)
def path_testregistry():
    """Fixture to generate absolute path for <repo>/test/registry."""
    path_registry = os.getenv("MLFLOW_REGISTRY_URI", "tests/registry")
    return pathlib.Path(path_registry).absolute()


@pytest.fixture(scope="session", params=os.listdir("tests/configurations"))
def config_file(request):
    """Fixture to provide each deepaas configuration path."""
    config_str = f"tests/configurations/{request.param}"
    return pathlib.Path(config_str).absolute()


@pytest.fixture(scope="module", name="tempdir")
def create_tempdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        yield tempdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(tempdir, path_testdata):
    """Fixture to copy the original data directory to the temp directory."""
    shutil.copytree(path_testdata, f"{tempdir}/{api.config.DATA_URI}")


@pytest.fixture(scope="module", autouse=True)
def copytree_models(tempdir, path_testmodels):
    """Fixture to copy the original models directory to the temp directory."""
    shutil.copytree(path_testmodels, f"{tempdir}/{api.config.MODELS_URI}")


@pytest.fixture(scope="module", autouse=True)
def copytree_registry(tempdir, path_testregistry):
    """Fixture to copy the original regist directory to the temp directory."""
    path_registry = os.getenv("MLFLOW_REGISTRY_URI", "tests/registry")
    shutil.copytree(path_testregistry, f"{tempdir}/{path_registry}")


def generate_signature(names, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD):
    """Function to generate dynamically signatures."""
    parameters = [inspect.Parameter(name, kind) for name in names]
    return inspect.Signature(parameters=parameters)


def generate_fields_fixture(signature):
    """Function to generate dynamically fixtures with dynamic arguments."""
    def fixture_function(**options):  # fmt: skip
        return {k: v for k, v in options.items() if v is not None}
    fixture_function.__signature__ = signature
    return pytest.fixture(scope="module")(fixture_function)


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return get_metadata to assert properties."""
    return api.get_metadata()


# Generate and inject fixtures for predict arguments
fields_predict = api.schemas.PredArgsSchema().fields
signature = generate_signature(fields_predict.keys())
globals()["predict_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def predictions(predict_kwds):
    """Fixture to return predictions to assert properties."""
    return api.predict(**predict_kwds)


# Generate and inject fixtures for training arguments
fields_training = api.schemas.TrainArgsSchema().fields
signature = generate_signature(fields_training.keys())
globals()["training_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def training(training_kwds):
    """Fixture to return training to assert properties."""
    return api.train(**training_kwds)
