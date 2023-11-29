"""Script to visualize autoencoder model output from MNIST dataset.
"""
# pylint: disable=unused-import
import argparse
import gzip
import logging
import pathlib
import sys

import mlflow
import numpy as np
from matplotlib import pyplot as plt

from demo_advanced import config

logger = logging.getLogger(__name__)


# Type validators ---------------------------------------------------
def int_images(string_value):
    """Validator converter for integer values for values higher than 0."""
    value = int(string_value)
    if value <= 0:
        raise ValueError("Images length must be greater than 0")
    return value


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
    *["-i", "--images"],
    help="Number of images to print (default: %(default)s)",
    type=int_images,
    default=5,
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
    *["images_file"],
    help="Path to 'gz' raw images file with MNIST data.",
    type=pathlib.Path,
)


# Script command actions --------------------------------------------
def _run_command(model_name, images_file, **options):
    # Common operations
    logging.basicConfig(level=options["verbosity"])
    logger.debug("Visualizing autoencoder %s output", model_name)

    # Load model from mlflow registry
    logger.info("Loading autoencoder %s from mlflow registry", model_name)
    model_uri = f"models:/{model_name}/{options['version']}"
    logger.debug("Using model uri %s visualization", model_uri)
    model = mlflow.tensorflow.load_model(model_uri)

    # Load images file from gz images_file
    logger.info("Loading MNIST images from file %s", images_file)
    with gzip.open(images_file, "rb") as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
        images = images.reshape(-1, *config.IMAGES_SHAPE)

    # Collect images for visualization
    logger.info("Collecting MNIST images from file %s", images_file)
    rng = np.random.default_rng()
    indexes = rng.choice(len(images), size=options["images"], replace=False)
    logger.debug("Selected image indexes: %s", indexes)
    images = images[indexes]

    # Generate autoencoder output from images
    encoded = model.encoder(images)
    decoded = model.decoder(encoded)

    # Generate plot using random images
    logger.info("Generating plot with random MNIST image indexes")
    fig, axes = plt.subplots(nrows=options["images"], ncols=3)
    for row, raw, enc, dec in zip(axes, images, encoded, decoded):
        row[0].imshow(raw, cmap="gray")
        row[1].imshow((enc,), cmap="gray")
        row[2].imshow(dec, cmap="gray")

    # Display plot using tight layout
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()

    # End of program
    logger.info("End of raw MNIST images visualization script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
