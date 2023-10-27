"""Script to generate MNIST data encoder to reduce dimensionality of input
data.
"""
# pylint: disable=invalid-name
import argparse
import logging
import sys

import mlflow
import numpy as np
import tensorflow as tf
from keras import layers
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from deepaas_full import config

logger = logging.getLogger(__name__)
mlflow_client = mlflow.MlflowClient(config.MODELS_PATH)


# Type validators ---------------------------------------------------
def latent_dim(string_value):
    """Validator converter for integer values for values higher than 0."""
    value = int(string_value)
    if value <= 0:
        raise ValueError("Latent_dim must be greater than 0")
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
    *["--latent_dim"],
    help="Dimension of encoded output vector (default: %(default)s)",
    type=latent_dim,
    default=32,
)
parser.add_argument(
    *["-n", "--name"],
    help="Model name to use for identification on mlflow registry.",
    type=str,
    required=True,
)


# Script command actions --------------------------------------------
def _run_command(name, **options):
    # Common operations
    logging.basicConfig(level=options["verbosity"])
    logger.debug("Generating MNIST encoder Model as %s", name)

    # Generation of encoder model from command inputs
    logger.info("Encoder layers with %s latent_dim", options["latent_dim"])
    encoder = [
        layers.Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 1)),
        layers.Flatten(),
        layers.Dense(options["latent_dim"] * 4, activation="relu"),
        layers.Dense(options["latent_dim"] * 2, activation="relu"),
        layers.Dense(options["latent_dim"] * 1, activation="relu"),
    ]
    logger.debug("Encoder layers generated: %s", encoder)

    # Generation of decoder model from command inputs
    logger.info("Decoder layers with %s latent_dim", options["latent_dim"])
    decoder = [
        layers.Input(shape=(options["latent_dim"],)),
        layers.Dense(784, activation="sigmoid"),
        layers.Reshape((28, 28)),
    ]
    logger.debug("Decoder layers generated: %s", decoder)

    # Merge encoder and decoder into autoencoder
    logger.info("Merging encoder and autoencoder layers to autoencoder model")
    model = tf.keras.Sequential(encoder + decoder[1:])
    model.encoder = tf.keras.Sequential(encoder)
    model.decoder = tf.keras.Sequential(decoder)
    logger.debug("Autoencoder model generated: %s", model.summary())

    # Set model optimizer, loss and metrics
    logger.info("Compile using Adam optimizer and MeanAbsoluteError loss.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanAbsoluteError(),
    )

    # Create mlflow autoencoder signature
    logger.info("Generating autoencoder signature for mlflow")
    io_shape = (-1, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
    input_schema = Schema([TensorSpec(np.dtype(np.float64), io_shape)])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), io_shape)])
    signature = ModelSignature(input_schema, output_schema)
    logger.debug("Signature generated: %s", signature)

    # Saving model experiment to mlflow experiments
    logger.info("Saving autoencoder in mlflow experiments.")
    with mlflow.start_run():
        info = mlflow.tensorflow.log_model(model, "model", signature=signature)
    mlflow.register_model(info.model_uri, name)
    logger.debug("Autoencoder saved with details: %s", info)

    # End of program
    logger.info("End of MNIST autoencoder creation script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
