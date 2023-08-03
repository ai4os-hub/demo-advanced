"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest
from deepaas.model.v2.wrapper import UploadedFile

import api


@pytest.fixture(scope="module", params=["t100-images.npy"])
def input_file(request):
    """Fixture to provide the input_file argument to api.predict."""
    filepath = f"{api.config.DATA_PATH}/interim"
    return UploadedFile("", filename=f"{filepath}/{request.param}")


@pytest.fixture(scope="module", params=["test_simplemodel"])
def model_name(request):
    """Fixture to provide the model_name argument to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[1])
def version(request):
    """Fixture to provide the model version argument to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[None, 20])
def batch_size(request):
    """Fixture to provide the batch_size option to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[None, 2])
def steps(request):
    """Fixture to provide the steps option to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=["application/json"])
def accept(request):
    """Fixture to provide the accept argument to api.predict."""
    return request.param
