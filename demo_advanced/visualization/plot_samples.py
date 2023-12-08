"""Script to visualize raw samples from MNIST dataset.
"""
# pylint: disable=unused-import
import argparse
import gzip
import logging
import math
import pathlib
import sys

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
    logging.basicConfig(level=options["verbosity"])

    # Load images file from gz images_file
    images_file = f"{config.DATA_URI}/raw/{images_file}.gz"
    logger.info("Loading MNIST images from file %s", images_file)
    with gzip.open(images_file, "rb") as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
        images = images.reshape(-1, *config.IMAGES_SHAPE)

    # Load labels file from gz labels_file
    labels_file = f"{config.DATA_URI}/raw/{labels_file}.gz"
    logger.info("Loading MNIST labels from file %s", labels_file)
    with gzip.open(labels_file, "rb") as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)

    # Generate plot using random images
    logger.info("Generating plot with random MNIST image indexes")
    nrows = math.isqrt(options["images"])
    fig, axs = plt.subplots(nrows, nrows + 1)
    for ax, _ in zip(axs.flat, range(options["images"])):
        i = np.random.randint(0, len(images))
        ax.imshow(images[i])
        ax.set_title(labels[i])

    # Display plot using tight layout
    fig.tight_layout()
    plt.show()

    # End of program
    logger.info("End of raw MNIST images visualization script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
