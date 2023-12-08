"""Fixtures module for api training. This is a configuration file designed
to prepare the tests function arguments on the test_*.py files located in
the same folder.

You can add new fixtures following the next structure:
```py
@pytest.fixture(scope="module", params=[{list of possible arguments}])
def argument_name(request):
    # You can add setup code here for your argument/fixture
    return request.param  # Argument that will be passed to the test
```
The fixture argument `request` includes the parameter generated by the
`params` list. Every test in the folder that uses the fixture will be run
at least once with each of the values inside `params` list unless specified
otherwise. The parameter is stored inside `request.param`.

When multiple fixtures are defined with more than one parameter, every tests
will run multiple times, each with one of all the possible combinations of
the generated parameters unless specified otherwise. For example, in the
following configuration:
```py
@pytest.fixture(scope="module", params=['a','b'])
def my_fixture1(request):
    return request.param

@pytest.fixture(scope="module", params=['x','y'])
def my_fixture2(request):
    return request.param
```
The for the test functions in this folder, the following combinations will
be generated:
    - Tests that use only one my_fixture1: ['a','b']
    - Tests that use only one my_fixture2: ['x','y']
    - Tests that use both: [('a','x'), ('a','y'), ('b','x'), ('b','y')]
    - Tests that use none of the fixtures: []

Be careful when using multiple fixtures with multiple parameters, as the
number of tests generated can grow exponentially.
"""
# pylint: disable=redefined-outer-name
import pytest

import api


@pytest.fixture(scope="module", params=["t100-dataset.npz"])
def input_file(request):
    """Fixture to provide the dataset argument to api.train."""
    return f"{api.config.DATA_URI}/processed/{request.param}"


@pytest.fixture(scope="module", params=["simple_convolution"])
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
