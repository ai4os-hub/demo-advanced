"""Module for defining custom web fields to use on the API interface.
"""
from webargs import fields, validate
import marshmallow

from . import parsers


# TODO: add/edit/remove arguments for predictions
class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    # Keep order of the parameters as they are defined.
    class Meta:
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    input_file = fields.Field(
        metadata={
            "description": "Custom file to generate predictions.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    argument_1 = fields.Integer(
        metadata={
            "description": "Custom required argument for predictions.",
        },
        required=True,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    option_1 = fields.Integer(
        metadata={
            "description": "Custom optional argument for predictions.",
        },
        required=False,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    option_2 = fields.Integer(
        metadata={
            "description": "Custom optional argument for predictions.",
        },
        required=False,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(parsers.content_types),
    )


# TODO: add/edit/remove arguments for predictions
class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    # Keep order of the parameters as they are defined.
    class Meta:
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    input_file = fields.Field(
        metadata={
            "description": "Custom file to use for model training.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    argument_1 = fields.Integer(
        metadata={
            "description": "Custom required argument for training.",
        },
        required=True,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    option_1 = fields.Integer(
        metadata={
            "description": "Custom optional argument for training.",
        },
        required=False,
    )

    # TODO: XXXXXX Complete description XXXXXXXXXXX.
    option_2 = fields.Integer(
        metadata={
            "description": "Custom optional argument for training.",
        },
        required=False,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(parsers.content_types),
    )
