"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging

import mlflow
from aiohttp.web import HTTPException

import deepaas_full

from . import config, parsers, schemas, utils

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
            "models": utils.ls_models(),
            "datasets": utils.ls_datasets(),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(model_uri, input_file, accept, **options):
    """Performs {model} prediction from given input data and parameters.

    Arguments:
        model_uri -- Model URI from MLFlow to use for prediction values.
        input_file -- NPY file with images to perform predictions from model.
        accept -- Response parser type.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Options:
        batch_size -- Number of samples per batch.
        steps -- Steps before prediction round is finished.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values or files.
    """
    try:
        logger.debug("input_file: %s", input_file)
        logger.debug("options: %s", options)
        model = mlflow.tensorflow.load_model(model_uri)
        result = deepaas_full.predict(model, input_file.filename, **options)
        logger.debug("accept: %s", accept)
        return parsers.response_parsers[accept](result)
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(model_uri, input_file, **options):
    """Performs {model} training from given input data and parameters.

    Arguments:
        model_uri -- Model URI from MLFlow to use for training.
        input_file -- NPZ file with images and labels to use for training.
        **options -- Arbitrary keyword arguments from TrainArgsSchema.

    Options:
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
        logger.debug("input_file: %s", input_file)
        logger.debug("options: %s", options)
        model = mlflow.tensorflow.load_model(model_uri)
        with mlflow.start_run(nested=False) as run:
            mlflow.tensorflow.autolog()
            deepaas_full.training(model, input_file, **options)
        return dict(mlflow.get_run(run.info.run_id).info)
    except Exception as err:
        raise HTTPException(reason=err) from err
