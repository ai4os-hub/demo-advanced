"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest

import api


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
