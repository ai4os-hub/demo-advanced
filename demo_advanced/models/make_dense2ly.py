"""Script to generate a MNIST sequential model with 2 dense layers of neurons.
"""
# pylint: disable=unused-import
import argparse
import logging
import sys

import tensorflow as tf
from keras import layers

from demo_advanced import config

logger = logging.getLogger(__name__)


# Type validators ---------------------------------------------------
def input_len(string_value):
    """Validator converter for integer values for values higher than 0."""
    value = int(string_value)
    if value <= 0:
        raise ValueError("Input length must be greater than 0")
    return value


def learning_rate(string_value):
    """Validator converter for float values for values between 1 and 0."""
    value = float(string_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Learning_rate factor must be between 0.0 and 1.0")
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
    *["--input_len"],
    help="Dimension of encoded input vectors (default: %(default)s)",
    type=input_len,
    default=32,
)
parser.add_argument(
    *["--learning_rate"],
    help="Weights rate respect to loss gradient (default: %(default)s)",
    type=learning_rate,
    default=1e-3,
)
parser.add_argument(
    *["-n", "--name"],
    help="Model name to use for identification on models folder.",
    type=str,
    required=True,
)


# Script command actions --------------------------------------------
def _run_command(name, **options):
    # Common operations
    logging.basicConfig(level=options["verbosity"])
    logger.debug("Generating MNIST convolution Model as %s", name)

    # Generation of the model from command inputs
    logger.info("Generating MNIST model with dense layers from configuration")
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(options["input_len"],)),
            layers.Dense(128, activation="relu"),
            layers.Dense(config.LABEL_DIMENSIONS, activation="softmax"),
        ]
    )
    logger.debug("Model generated: %s", model.summary())

    # Set model optimizer, loss and metrics
    logger.info("Compile with learning_rate : %s", options["learning_rate"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(options["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    # Saving model to models folder
    logger.info("Saving model in %s.", config.MODELS_URI)
    save_path = f"{config.MODELS_URI}/{name}"
    model.save(save_path)
    logger.debug("Model saved with details: %s", save_path)

    # End of program
    logger.info("End of MNIST model creation script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
