"""Functions to integrate {} model with the DEEPaaS API. 
TODO: Add model name from template

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging

from api import config, fields, parsers, utils

logger = logging.getLogger("__name__")


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
        "extra_1": utils.ls_local("models_folder1"),  # TODO: Replace names
        "extra_2": utils.ls_local("models_folder2"),  # TODO: Replace names
    }
    logger.debug("Package model metadata: %d", metadata)
    return metadata


def get_predict_args():
    """Return the arguments that are needed to perform a prediction.

    Returns:
        Dictionary of webargs fields.
    """
    predict_args = {
        "input_file": fields.CustomFileArg(required=True),
        "argument_1": fields.CustomPredArg1(required=True),
        "option_1": fields.CustomPredArg2(required=False),
        "option_2": fields.CustomPredArg2(required=False),
    }
    logger.debug("Web arguments: %d", predict_args)
    return predict_args


def get_train_args():
    """Return the arguments that are needed to perform a training.

    Returns:
        Dictionary of webargs fields.
    """
    predict_args = {
        "input_file": fields.CustomFileArg(required=True),
        "argument_1": fields.CustomTrainArg1(required=True),
        "option_1": fields.CustomTrainArg2(required=False),
        "option_2": fields.CustomTrainArg2(required=False),
    }
    logger.debug("Web arguments: %d", predict_args)
    return predict_args


def predict(input_file, argument_1, **options):
    """Performs {model} prediction from given input data and parameters.

    Args:
        input_file: Input data to generate prediction values.
        argument_1: Required argument 1 to generate prediction values.
        **options: Arbitrary keyword arguments from get_predict_args.

    Options:
        option_1: Optional argument 1 to generate prediction values.
        option_2: Optional argument 2 to generate prediction values.

    Returns:
        The predicted model values or files.
    """
    logger.debug("input_file: %d, argument_1: %d", input_file, argument_1)
    logger.debug("Options: %d", options)
    raise NotImplementedError  # TODO: Replace by model predict function
    return parsers.response_parsers[accept](*result)


def train(input_file, argument_1, **options):
    """Performs {model} training from given input data and parameters.

    Args:
        input_file: Input data to perform model training.
        argument_1: Required argument 1 to perform model training.
        **options: Arbitrary keyword arguments from get_train_args.

    Options:
        option_1: Optional argument 1 to perform model training.
        option_2: Optional argument 2 to perform model training.

    Returns:
        The train result values or files.
    """
    logger.debug("input_file: %d, argument_1: %d", input_file, argument_1)
    logger.debug("Options: %d", options)
    raise NotImplementedError  # TODO: Replace by model train function
    return parsers.response_parsers[accept](*result)
