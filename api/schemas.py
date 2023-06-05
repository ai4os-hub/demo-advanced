"""Module for defining custom web fields to use on the API interface.
"""
import marshmallow
import mlflow
from webargs import ValidationError, fields, validate

from . import config, parsers, utils


class ModelURI(fields.String):
    """Field that takes a string and validates against the available models
    at the MLFlow instance connected. ModelNameVersion requires the format
    'models:/{model_name}/{version}' to find it at the MLFlow repository.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        try:  # add cache: https://github.com/mlflow/mlflow/issues/3123
            mlflow.tensorflow.load_model(value)
            return value
        except mlflow.MlflowException as err:
            raise ValidationError(err.message) from err
        except OSError as err:
            raise ValidationError(f"Wrong mlflow model uri {value}") from err


class Dataset(fields.String):
    """Field that takes a string and validates against current available
    data files at config.DATA_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_datasets():
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / value)


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model_uri = ModelURI(
        metadata={
            "description": "String 'models:/name/version' from MLFlow models.",
        },
        required=True,
    )

    input_file = fields.Field(
        metadata={
            "description": "NPY file with images data for predictions.",
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

    model_uri = ModelURI(
        metadata={
            "description": "Str 'models:/name/version' from MLFlow models.",
        },
        required=True,
    )

    input_file = Dataset(
        metadata={
            "description": "NPZ dataset from metadata for training input.",
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
