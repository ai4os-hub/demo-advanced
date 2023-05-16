"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest

from api import api_v1 as api


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return defined api metadata."""
    return api.get_metadata()


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def predict_input1(request):  # TODO: Rename function
    """Fixture to provide the first input argument to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def predict_input2(request):  # TODO: Rename function
    """Fixture to provide the second input argument to api.predict."""
    return request.param


@pytest.fixture(scope="module")
def predict_args(predict_input1, predict_input2):
    """Fixture to return positional arguments for predictions."""
    return (predict_input1, predict_input2)


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def predict_option1(request):  # TODO: Rename function
    """Fixture to provide the first input option to api.predict."""
    return request.param


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def predict_option2(request):  # TODO: Rename function
    """Fixture to provide the second input option to api.predict."""
    return request.param


@pytest.fixture(scope="module")
def predict_kwds(predict_option1, predict_option2):
    """Fixture to return arbitrary keyword arguments for predictions."""
    keys = [k for k, v in api.get_predict_args().items() if not v.required]
    pred_kwds = {}  # TODO: Customize/Complete with predict options
    pred_kwds[keys[0]] = predict_option1
    pred_kwds[keys[1]] = predict_option2
    return {k: v for k, v in pred_kwds.items() if v is not None}


@pytest.fixture(scope="module")
def predictions(predict_args, predict_kwds):
    """Fixture to return predictions to assert properties."""
    return api.predict(*predict_args, **predict_kwds)


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def train_input1(request):  # TODO: Rename function
    """Fixture to provide the first input argument to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def train_input2(request):  # TODO: Rename function
    """Fixture to provide the second input argument to api.train."""
    return request.param


@pytest.fixture(scope="module")
def train_args(train_input1, train_input2):
    """Fixture to return positional arguments for training."""
    return (train_input1, train_input2)


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def train_option1(request):  # TODO: Rename function
    """Fixture to provide the first input option to api.train."""
    return request.param


@pytest.fixture(scope="module", params=[])  # TODO: Add your parameters
def train_option2(request):  # TODO: Rename function
    """Fixture to provide the second input option to api.train."""
    return request.param


@pytest.fixture(scope="module")
def train_kwds(train_option1, train_option2):
    """Fixture to return arbitrary keyword arguments for training."""
    keys = [k for k, v in api.get_train_args().items() if not v.required]
    train_kwds = {}  # TODO: Customize/Complete with predict options
    train_kwds[keys[0]] = train_option1
    train_kwds[keys[1]] = train_option2
    return {k: v for k, v in train_kwds.items() if v is not None}


@pytest.fixture(scope="module")
def training(train_args, train_kwds):
    """Fixture to perform and return training to assert properties."""
    return api.train(*train_args, **train_kwds)
