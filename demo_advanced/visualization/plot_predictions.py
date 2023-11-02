"""Script to visualize model predictions from MNIST inputs.
"""
import argparse
import gzip
import logging
import pathlib
import sys

import mlflow
import numpy as np
import tensorflow as tf
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
    *["--encoder"],
    help="Autoencoder name to identify the encoder on mlflow registry.",
    type=str,
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
    logger.debug("Visualizing model %s predictions", model_name)

    # Load model from mlflow registry
    logger.info("Loading model %s from mlflow registry", model_name)
    model_uri = f"models:/{model_name}/{options['version']}"
    logger.debug("Using model uri %s visualization", model_uri)
    model = mlflow.tensorflow.load_model(model_uri)

    # Load encoder from mlflow registry if defined
    if options["encoder"] is not None:
        logger.info("Loading encoder %s from registry", options["encoder"])
        encoder_uri = f"models:/{options['encoder']}/Production"
        logger.debug("Using model uri %s for encoding", encoder_uri)
        encoder = mlflow.tensorflow.load_model(encoder_uri).encoder
    else:
        encoder = None

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

    # Collect images and labels for visualization
    logger.info("Collecting MNIST images and labels from data files")
    rng = np.random.default_rng()
    indexes = rng.choice(len(images), size=options["images"], replace=False)
    logger.debug("Selected image indexes: %s", indexes)
    images = images[indexes]
    labels = labels[indexes]

    # Generate model input from images
    logging.info("Generating model input from images")
    inputs = encoder(images) if encoder else images
    logging.debug("Model input shape: %s", inputs.shape)

    # Generate model output from input
    logging.info("Generating model output from input")
    outputs = model.predict(inputs)
    logging.debug("Model output shape: %s", outputs.shape)

    # Generate plot using random images
    logger.info("Generating plot with random MNIST image indexes")
    ncols = 3 if encoder else 2
    fig, axes = plt.subplots(options["images"], ncols)
    for row, raw, inp, out, lab in zip(axes, images, inputs, outputs, labels):
        row[0].imshow(raw, cmap="gray")
        if encoder:
            row[1].imshow((inp,), cmap="gray")
        row[-1].imshow((out, lab), cmap="gray")

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
