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
        "extra_1": utils.ls_local("models_folder1"),  # TODO: Replace names
        "extra_2": utils.ls_local("models_folder2"),  # TODO: Replace names
    }
    logger.debug("Package model metadata: %d", metadata)
    return metadata


# @utils.predict_arguments(schema=schemas.PredArgsSchema)
# def predict(input_file, argument_1, accept, **options):
#     """Performs {model} prediction from given input data and parameters.

#     Args:
#         input_file: Input data to generate prediction values.
#         argument_1: Required argument 1 to generate prediction values.
#         **options: Arbitrary keyword arguments from get_predict_args.

#     Options:
#         option_1: Optional argument 1 to generate prediction values.
#         option_2: Optional argument 2 to generate prediction values.

#     Returns:
#         The predicted model values or files.
#     """
#     logger.debug("input_file: %d, argument_1: %d", input_file, argument_1)
#     logger.debug("Options: %d", options)
#     raise NotImplementedError  # TODO: Replace by model predict function
#     # return parsers.response_parsers[accept](*result)


# @utils.train_arguments(schema=schemas.TrainArgsSchema)
# def train(input_file, target_file, **options):
@utils.predict_arguments(schema=schemas.TrainArgsSchema)
def predict(input_file, target_file, accept, **options):
    """Performs {model} training from given input data and parameters.

    Args:
        input_file: Input data file to perform model training.
        input_file: Input labels to file fit model training.
        **options: Arbitrary keyword arguments from get_train_args.

    Options:
        option_1: Optional argument 1 to perform model training.
        option_2: Optional argument 2 to perform model training.

    Returns:
        The train result values or files.
    """
    logger.debug("input_file: %d, target_file: %d", input_file, target_file)
    logger.debug("Options: %d", options)
    model = deepaas_full.create_model()
    input_file, target_file = input_file.filename, target_file.filename
    return deepaas_full.training(model, input_file, target_file, **options)
    # return parsers.response_parsers[accept](*result)
