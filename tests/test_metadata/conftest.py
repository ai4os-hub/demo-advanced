"""Module for evolution requirements fixtures."""
# pylint: disable=redefined-outer-name
import pytest

import api


@pytest.fixture(scope="module")
def metadata():
    """Fixture to return defined api metadata."""
    return api.get_metadata()
