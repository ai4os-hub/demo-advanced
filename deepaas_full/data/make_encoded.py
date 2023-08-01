"""Script to perform MNIST data encoding to prepare it for model
training.
"""
import argparse
import gzip
import logging
import pathlib
import sys

import numpy as np
import tensorflow as tf
import mlflow

from deepaas_full import config

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
    *["-o", "--output"],
    help="Output folder for skimmed files (default: %(default)s)",
    type=pathlib.Path,
    default="training_data.npz",
)
parser.add_argument(
    *["model_name"],
    help="Autoencoder name to use for identification on mlflow registry.",
    type=str,
)
parser.add_argument(
    *["--version"],
    help="Model version for model in model_name (default: %(default)s).",
    type=version,
    default="Production",
)
parser.add_argument(
    *["images_file"],
    help="Path to 'gz' raw images file with MNIST data.",
    type=pathlib.Path,
)
parser.add_argument(
    *["labels_file"],
    help="Path to 'gz' labels file classifying images file.",
    type=pathlib.Path,
)


# Script command actions --------------------------------------------
def _run_command(model_name, images_file, labels_file, **options):
    # Common operations
    logging.basicConfig(level=options["verbosity"])
    logger.debug("Encoding MNIST using %s autoencoder", model_name)

    # Load images file from gz images_file
    logger.info("Loading MNIST images from file %s", images_file)
    with gzip.open(images_file, "rb") as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
        images = images.reshape(-1, *config.IMAGES_SHAPE)
    images = images / 255.0

    # Load labels file from gz labels_file
    logger.info("Loading MNIST labels from file %s", labels_file)
    with gzip.open(labels_file, "rb") as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    labels = tf.keras.utils.to_categorical(labels, config.LABEL_DIMENSIONS)

    # Load model from mlflow registry
    logger.info("Loading autoencoder %s from mlflow registry", model_name)
    model_uri = f"models:/{model_name}/{options['version']}"
    logger.debug("Using model uri %s visualization", model_uri)
    model = mlflow.tensorflow.load_model(model_uri)

    # Process images using autoencoder encoder
    logger.info("Encoding MNIST images using %s autoencoder", model_name)
    encoded = model.encoder.predict(images)

    # Merge and save data in output file
    logger.info("Saving MNIST pre-process output at %s", options["output"])
    np.savez(options["output"], x_train=encoded, y_train=labels)

    # End of program
    logger.info("End of MNIST image processing script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
