"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging

from aiohttp.web import HTTPException

import deepaas_full as aimodel
from deepaas_full import config

from . import parsers, schemas, utils

logger = logging.getLogger(__name__)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:
        metadata = {
            "authors": [config.MODEL_METADATA.get("author_email")],
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            "datasets": utils.ls_datasets(),
            "models": utils.ls_models(),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(model_name, input_file, accept, **options):
    """Performs {model} prediction from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for prediction values.
        input_file -- NPY file with data to perform predictions from model.
        accept -- Response parser type.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Options:
        version -- MLFLow model version to use for predictions.
        batch_size -- Number of samples per batch.
        steps -- Steps before prediction round is finished.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values or files.
    """
    try:
        result = aimodel.predict(input_file.filename, model_name, **options)
        logger.debug("Using parser for: %s", accept)
        return parsers.response_parsers[accept](result)
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(model_name, input_file, accept, **options):
    """Performs {model} training from given input data and parameters.

    Arguments:
        model_name -- Model name from registry to use for training values.
        input_file -- NPZ file with data and labels to use for training.
        accept -- Response parser type.
        **options -- Arbitrary keyword arguments from TrainArgsSchema.

    Options:
        version -- MLFLow model version to use for predictions.
        epochs -- Number of epochs to train the model.
        initial_epoch -- Epoch at which to start training.
        steps_per_epoch -- Steps before declaring an epoch finished.
        shuffle -- Shuffle the training data before each epoch.
        validation_split -- Fraction of the data to be used as validation.
        validation_steps -- Steps to draw before stopping on validation.
        validation_batch_size -- Number of samples per validation batch.
        validation_freq -- Training epochs to run before validation.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        Dictionary containing mlflow run information.
    """
    try:
        result = aimodel.training(input_file, model_name, **options)
        # return parsers.response_parsers[accept](result)
        return result
    except Exception as err:
        raise HTTPException(reason=err) from err
