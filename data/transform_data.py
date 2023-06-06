"""Script to perform MNIST data pre-processing to prepare it for model
training.
"""
import argparse
import gzip
import logging
import pathlib
import sys

import numpy as np
import tensorflow as tf

LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LABEL_DIMENSIONS = 10
IMAGES_SHAPE = (-1, 28, 28)

logger = logging.getLogger(__name__)


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
    "images_file",
    help="Path to 'gz' raw images file with MNIST data.",
    type=pathlib.Path,
)
parser.add_argument(
    "labels_file",
    help="Path to 'gz' labels file classifying images file.",
    type=pathlib.Path,
)


# Script command actions --------------------------------------------
def _run_command(images_file, labels_file, **options):
    # Common operations
    logging.basicConfig(format=LOGGER_FORMAT, level=options["verbosity"])
    logger.info("Processing MNIST images at %s", options["output"])

    # Load images file from gz images_file
    logger.debug("Loading MNIST images from file %s", images_file)
    with gzip.open(images_file, "rb") as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
        images = images.reshape(*IMAGES_SHAPE)
    images = images / 255

    # Load labels file from gz labels_file
    logger.debug("Loading MNIST labels from file %s", labels_file)
    with gzip.open(labels_file, "rb") as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    labels = tf.keras.utils.to_categorical(labels, LABEL_DIMENSIONS)

    # Merge and save data in output file
    logger.debug("Saving MNIST pre-process output at %s", options["output"])
    np.savez(options["output"], images=images, labels=labels)

    # End of program
    logger.info("End of MNIST image processing script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
