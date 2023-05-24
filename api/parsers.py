"""Module for defining custom API response parsers and content types.
"""
import logging

logger = logging.getLogger(__name__)


def json_response(argument_1, *extra_values):
    """Converts the prediction or training results into json return format.

    Arguments:
        argument_1 -- _description_   TODO: Complete description
        extra_values -- _description_ TODO: Complete description

    Returns:
        Converted values into json dictionary format.
    """
    logger.debug("Response argument_1: %d", argument_1)
    raise NotImplementedError  # TODO: Convert result values into json dict
    # return json_dictionary


def pdf_response(argument_1, *extra_values):
    """Converts the prediction or training results into json return format.

    Arguments:
        argument_1 -- _description_   TODO: Complete description
        extra_values -- _description_ TODO: Complete description

    Returns:
        Converted values into pdf buffer format.
    """
    logger.debug("Response argument_1: %d", argument_1)
    raise NotImplementedError  # TODO: Convert result values into pdf buffer
    # buffer_out.name = "results_file.pdf"
    # return buffer_out


response_parsers = {
    "application/json": json_response,
    "application/pdf": pdf_response,
}
content_types = list(response_parsers)
