"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest
from deepaas.model.v2.wrapper import UploadedFile

import api


@pytest.fixture(scope="module", params=["models:/deepaas-full-testing/1"])
def model_uri(request):
    """Fixture to provide the model_uri argument to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=["test-images-idx3-ubyte.gz"])
def input_file(request):
    """Fixture to provide the input_file argument to api.predict."""
    return UploadedFile("", filename=api.config.DATA_PATH / request.param)


@pytest.fixture(scope="module", params=["application/json"])
def accept(request):
    """Fixture to provide the accept argument to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[None, 20])
def batch_size(request):
    """Fixture to provide the batch_size option to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[None, 2])
def steps(request):
    """Fixture to provide the steps option to api.predict."""
    return request.param


@pytest.fixture(scope="module")
def options(batch_size, steps):
    """Fixture to return arbitrary keyword options for predictions."""
    options = {}  # Customize/Complete with predict options
    options["batch_size"] = batch_size
    options["steps"] = steps
    return {k: v for k, v in options.items() if v is not None}


@pytest.fixture(scope="module")
def predictions(model_uri, input_file, accept, options):
    """Fixture to return predictions to assert properties."""
    return api.predict(model_uri, input_file, accept, **options)
