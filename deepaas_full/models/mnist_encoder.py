"""Script to generate MNIST data encoder to reduce dimensionality of input
data.
"""
import argparse
import logging
import sys

import mlflow
import numpy as np
import tensorflow as tf
from keras import layers
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

from deepaas_full import config

logger = logging.getLogger(__name__)
mlflow_client = MlflowClient()


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
    default=64,
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
    logger.info("Generating MNIST encoder Model as %s", name)

    # Generation of the model from command inputs
    encoder = tf.keras.Sequential(
        [
            layers.Flatten(input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE)),
            layers.Dense(options["latent_dim"], activation="relu"),
        ]
    )
    logger.info("Encoder generated: %s", encoder.summary())

    # Set model optimizer, loss and metrics
    logger.info("Compile using Adam optimizer and MeanAbsoluteError loss.")
    encoder.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanAbsoluteError(),
    )

    # Create mlflow signature
    input_shape = (-1, config.IMAGE_SIZE, config.IMAGE_SIZE, 1)
    input_schema = Schema([TensorSpec(np.dtype(np.float64), input_shape)])
    output_shape = (-1, options["latent_dim"])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), output_shape)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    # Saving model experiment to mlflow experiments
    logger.info("Saving model in mlflow experiments.")
    with mlflow.start_run():
        i = mlflow.tensorflow.log_model(encoder, "model", signature=signature)
    mlflow.register_model(i.model_uri, name)

    logger.info("Creating encoder %s in mlflow model registry", name)
    mlflow_client.create_registered_model(name)

    # End of program
    logger.info("End of MNIST encoder creation script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
