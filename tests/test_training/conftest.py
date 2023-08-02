"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest

import api


@pytest.fixture(scope="module", params=["t100-dataset.npz"])
def input_file(request):
    """Fixture to provide the dataset argument to api.train."""
    return api.config.DATA_PATH / "processed" / request.param


@pytest.fixture(scope="module", params=["test_simplemodel"])
def model_name(request):
    """Fixture to provide the model_name argument to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[1])
def version(request):
    """Fixture to provide the model version argument to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[2])
def epochs(request):
    """Fixture to provide the epochs option to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[None, 1])
def initial_epoch(request):
    """Fixture to provide the initial_epoch option to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[None, False])
def shuffle(request):
    """Fixture to provide the shuffle option to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[None, 0.1])
def validation_split(request):
    """Fixture to provide the validation_split option to api.train."""
    return request.param


@pytest.fixture(scope="module", params=["application/json"])
def accept(request):
    """Fixture to provide the accept argument to api.predict."""
    return request.param
