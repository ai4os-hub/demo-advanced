"""Script to train a MNIST model with a dataset.
"""
# pylint: disable=unused-import
import argparse
import logging
import pathlib
import pprint
import sys

import demo_advanced as aimodel
from demo_advanced import config  # noqa: F401

logger = logging.getLogger(__name__)


# Type validators ---------------------------------------------------
def version(string_value):
    """Validator converter for version values and/or digits."""
    if string_value in ["Staging", "Production"]:
        return string_value
    if not string_value.isdigit():
        raise ValueError("Version must be int, 'Staging' or 'Production'")
    value = int(string_value)
    if value <= 0:
        raise ValueError("Version number must be greater than 0")
    return value


def epochs(string_value):
    """Validator converter for integer values for values higher than 0."""
    value = int(string_value)
    if value <= 0:
        raise ValueError("Epochs must be greater than 0")
    return value


def validation_split(string_value):
    """Validator converter for float values for values between 1 and 0."""
    value = float(string_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Validation split factor must be between 0.0 and 1.0")
    return value


# Script arguments definition ---------------------------------------
parser = argparse.ArgumentParser(
    prog="PROG",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="See '<command> --help' to read about a specific sub-command.",
)
parser.add_argument(
    *["-v", "--verbosity"],
    help="Sets the logging level (default: %(default)s)",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
)
parser.add_argument(
    *["model_name"],
    help="Model name to use for identification from models folder.",
    type=str,
)
parser.add_argument(
    *["--version"],
    help="Model version for model in model_name (default: %(default)s).",
    type=version,
    default="Production",
)
parser.add_argument(
    *["input_file"],
    help="Dataset NPZ file to train the model.",
    type=pathlib.Path,
)
parser.add_argument(
    *["--epochs"],
    help="Number of epochs to train the model (default: %(default)s).",
    type=epochs,
    default=6,
)
parser.add_argument(
    *["--validation_split"],
    help="Data fraction to use as validation (default: %(default)s).",
    type=validation_split,
    default=0.1,
)


# Script command actions --------------------------------------------
def _run_command(model_name, input_file, **options):
    # Common operations
    logging.basicConfig(level=options.pop("verbosity"))
    logger.debug("Training new %s model", model_name)

    # Get training file from data directory
    input_file = f"{config.DATA_URI}/processed/{input_file}.npz"
    logger.debug("Loading data from input_file: %s", input_file)

    # Call training function from aimodel
    logger.info("Train model using options: %s", options)
    model_uri = pathlib.Path(config.MODELS_URI) / model_name
    result = aimodel.train(model_uri, input_file, **options)

    # End of program
    logger.info("End of MNIST model training script")
    pprint.pprint(dict(result))


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
