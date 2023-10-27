"""Script to predict using MNIST model with an input file.
"""
import argparse
import logging
import pathlib
import sys

import mlflow
import numpy as np

import deepaas_full as aimodel
from deepaas_full import config

logger = logging.getLogger(__name__)
mlflow_client = mlflow.MlflowClient(config.MODELS_PATH)


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


def batch_size(string_value):
    """Validator converter for integer values for values higher than 0."""
    value = int(string_value)
    if value <= 0:
        raise ValueError("Batch size must be greater than 0")
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
    help="Model name to use for identification on mlflow registry.",
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
    help="NPY file to generate model predictions.",
    type=pathlib.Path,
)
parser.add_argument(
    *["--batch_size"],
    help="Number of samples per batch (default: %(default)s).",
    type=batch_size,
)
parser.add_argument(
    *["output_file"],
    help="NPY file where to write model predictions.",
    type=pathlib.Path,
)


# Script command actions --------------------------------------------
def _run_command(model_name, input_file, output_file, **options):
    # Common operations
    logging.basicConfig(level=options.pop("verbosity"))
    logger.debug("Predict using %s model", model_name)

    # Call training function from aimodel
    logger.info("Generate predictions with options: %s", options)
    result = aimodel.predict(input_file, model_name, **options)

    # Write predictions into output file
    logger.info("Writing predictions to output file %s", output_file)
    np.save(output_file, result)

    # End of program
    logger.info("End of MNIST model training script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
