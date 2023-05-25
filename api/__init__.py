"""Endpoint functions to integrate your model with the DEEPaaS API. 

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging

from . import config, schemas, parsers, utils
import deepaas_full

logger = logging.getLogger(__name__)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        "authors": config.MODEL_METADATA.get("author"),
        "description": config.MODEL_METADATA.get("summary"),
        "license": config.MODEL_METADATA.get("license"),
        "version": config.MODEL_METADATA.get("version"),
        "checkpoints": utils.ls_models(),
    }
    logger.debug("Package model metadata: %d", metadata)
    return metadata


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(model, input_file, accept, **options):
    """Performs {model} prediction from given input data and parameters.

    Arguments:
        model -- Model from checkpoint to use for predicting values.
        input_file -- Input data file to perform predictions from model.
        accept -- Response parser type.
        **options -- Arbitrary keyword arguments from PredArgsSchema.

    Options:
        batch_size -- Number of samples per batch.
        steps -- Steps before prediction round is finished.

    Returns:
        The predicted model values or files.
    """
    logger.debug("input_file: %s", input_file)
    logger.debug("options: %d", options)
    result = deepaas_full.predict(model, input_file.filename, **options)
    logger.debug("accept: %s", accept)
    return parsers.response_parsers[accept](*result)


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(model, input_file, target_file, accept, **options):
    """Performs {model} training from given input data and parameters.

    Arguments:
        model -- Model from checkpoint to train with the input files.
        input_file -- Input data file to perform model training.
        target_file -- Input labels to file fit model training.
        accept -- Response parser type.
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

    Returns:
        Parsed history/summary of the training process.
    """
    logger.debug("input_file: %s, target_file: %s", input_file, target_file)
    input_files = input_file.filename, target_file.filename
    logger.debug("options: %d", options)
    result = deepaas_full.training(model, *input_files, **options)
    logger.debug("accept: %s", accept)
    return parsers.response_parsers[accept](*result)
