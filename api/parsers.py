"""Module for defining custom API response parsers and content types.
"""
import logging
import builtins
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def json_response(values, **extra_values):
    """Converts the prediction or training results into json return format.

    Arguments:
        values -- Result value from method call.
        extra_values -- Not used, added for illustration purpose.

    Returns:
        Converted values into json dictionary format.
    """
    logger.debug("Response result: %d", values)
    logger.debug("Response options: %d", extra_values)
    match type(values):
        case builtins.dict | builtins.list:
            return values
        case np.ndarray | tf.Tensor:
            return values.tolist()
        case _:
            return dict(values)


def pdf_response(values, **extra_values):
    """Converts the prediction or training results into json return format.

    Arguments:
        values -- Result value from method call.
        extra_values -- Not used, added for illustration purpose.

    Returns:
        Converted values into pdf buffer format.
    """
    logger.debug("Response result: %d", values)
    logger.debug("Response options: %d", extra_values)
    raise NotImplementedError


response_parsers = {
    "application/json": json_response,
    "application/pdf": pdf_response,
}
content_types = list(response_parsers)
