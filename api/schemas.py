"""Module for defining custom web fields to use on the API interface.
"""
import marshmallow
from webargs import ValidationError, fields, validate

import deepaas_full
from tensorflow import errors as tf_errors

from . import config, parsers


class Checkpoint(fields.String):
    """Field that takes a string and validates against current available
    models at config.MODELS_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        try:
            model = deepaas_full.create_model()
            model.load_weights(config.MODELS_PATH / value)
            return model
        except tf_errors.NotFoundError as err:
            raise ValidationError(f"Checkpoint `{value}` not found.") from err


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model = Checkpoint(
        metadata={
            "description": "Checkpoint to use for predictions.",
        },
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "Custom file to generate predictions.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    batch_size = fields.Integer(
        metadata={
            "description": "Number of samples per batch.",
        },
        required=False,
        validate=validate.Range(min=1),
    )

    steps = fields.Integer(
        metadata={
            "description": "Steps before prediction round is finished.",
        },
        required=False,
        validate=validate.Range(min=1),
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(parsers.content_types),
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model = Checkpoint(
        metadata={
            "description": "Checkpoint to use for training.",
        },
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "Custom file to use for model training.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    target_file = fields.Field(
        metadata={
            "description": "Custom file to use for model training.",
            "type": "file",
            "location": "form",
        },
        required=True,
    )

    epochs = fields.Integer(
        metadata={
            "description": "Number of epochs to train the model.",
        },
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )

    initial_epoch = fields.Integer(
        metadata={
            "description": "Epoch at which to start training.",
        },
        required=False,
        load_default=0,
        validate=validate.Range(min=0),
    )

    steps_per_epoch = fields.Integer(
        metadata={
            "description": "Steps before declaring an epoch finished.",
        },
        required=False,
        validate=validate.Range(min=0),
    )

    shuffle = fields.Boolean(
        metadata={
            "description": "Shuffle the training data before each epoch.",
        },
        required=False,
        load_default=True,
    )

    validation_split = fields.Float(
        metadata={
            "description": "Fraction of the data to be used as validation.",
        },
        required=False,
        load_default=0.0,
        validate=validate.Range(min=0.0, max=1.0),
    )

    validation_steps = fields.Integer(
        metadata={
            "description": "Steps to draw before stopping on validation.",
        },
        required=False,
        validate=validate.Range(min=0),
    )

    validation_batch_size = fields.Integer(
        metadata={
            "description": "Number of samples per validation batch.",
        },
        required=False,
        validate=validate.Range(min=0),
    )

    validation_freq = fields.Integer(
        metadata={
            "description": "Training epochs to run before validation.",
        },
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        required=True,
        validate=validate.OneOf(parsers.content_types),
    )
