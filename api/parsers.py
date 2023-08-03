"""Module for defining custom API response parsers and content types.
"""
import logging
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
    if isinstance(values, (dict, list)):
        return values
    if isinstance(values, (np.ndarray, tf.Tensor)):
        return values.tolist()
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
