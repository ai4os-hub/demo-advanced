"""Script to perform MNIST data pre-processing to prepare it for encoder
training.
"""
# pylint: disable=unused-import
import argparse
import gzip
import logging
import pathlib
import sys

import numpy as np

from demo_advanced import config

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


# Script command actions --------------------------------------------
def _run_command(images_file, **options):
    # Common operations
    logging.basicConfig(level=options["verbosity"])
    logger.debug("Processing MNIST images at %s", options["output"])

    # Load images file from gz images_file
    images_file = f"{config.DATA_URI}/raw/{images_file}.gz"
    logger.info("Loading MNIST images from file %s", images_file)
    with gzip.open(images_file, "rb") as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
        images = images.reshape(-1, *config.IMAGES_SHAPE)
    images = images / 255.0

    # Merge and save data in output file
    logger.info("Saving MNIST pre-process output at %s", options["output"])
    np.savez(options["output"], x_train=images, y_train=images)

    # End of program
    logger.info("End of MNIST image processing script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
