"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest

import api


@pytest.fixture(scope="module", params=["models:/deepaas_full-testing/1"])
def model_uri(request):
    """Fixture to provide the model_uri argument to api.train."""
    return request.param


@pytest.fixture(scope="module", params=["t100-dataset.npz"])
def dataset(request):
    """Fixture to provide the dataset argument to api.train."""
    return api.config.DATA_PATH / request.param


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


@pytest.fixture(scope="module")
def options(epochs, initial_epoch, shuffle, validation_split):
    """Fixture to return arbitrary keyword arguments for training."""
    options = {}  # Customize/Complete with training options
    options["epochs"] = epochs
    options["initial_epoch"] = initial_epoch
    options["shuffle"] = shuffle
    options["validation_split"] = validation_split
    return {k: v for k, v in options.items() if v is not None}


@pytest.fixture(scope="module")
def training(model_uri, dataset, options):
    """Fixture to perform and return training to assert properties."""
    return api.train(model_uri, dataset, **options)
