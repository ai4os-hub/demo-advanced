"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging
import time

import tensorflow as tf
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
            "checkpoints": utils.ls_models(),
            "datasets": utils.ls_datasets(),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(checkpoint, input_file, accept, **options):
    """Performs {model} prediction from given input data and parameters.

    Arguments:
        checkpoint -- Model from checkpoint to use for predicting values.
        input_file -- Input data file to perform predictions from model.
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
        model = tf.keras.models.load_model(checkpoint)
        result = deepaas_full.predict(model, input_file.filename, **options)
        logger.debug("accept: %s", accept)
        return parsers.response_parsers[accept](result)
    except Exception as err:
        raise HTTPException(reason=err) from err


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(checkpoint, inputs_ds, labels_ds, **options):
    """Performs {model} training from given input data and parameters.

    Arguments:
        checkpoint -- Model from checkpoint to train with the input files.
        inputs_ds -- Dataset file name to use as data for training.
        labels_ds -- Dataset file name to use as labels to fit model.
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
        Parsed history/summary of the training process.
    """
    try:
        logger.debug("inputs_ds: %s, labels_ds: %s", inputs_ds, labels_ds)
        ckpt_name = f"{time.strftime('%Y%m%d-%H%M%S')}.cp.ckpt"
        options["callbacks"] = utils.generate_callbacks(ckpt_name)
        logger.debug("options: %s", options)
        model = tf.keras.models.load_model(checkpoint)
        return deepaas_full.training(model, inputs_ds, labels_ds, **options)
    except Exception as err:
        raise HTTPException(reason=err) from err
