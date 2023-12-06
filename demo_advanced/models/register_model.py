"""Script to train a MNIST model with a dataset.
"""
# pylint: disable=unused-import
import argparse
import logging
import sys

import mlflow

from demo_advanced import config  # noqa: F401

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
    *["run_id"],
    help="Run id from mlflow experiment training.",
    type=str,
)
parser.add_argument(
    *["model_name"],
    help="Model name to use for identification on mlflow registry.",
    type=str,
)


# Script command actions --------------------------------------------
def _run_command(run_id, model_name, **options):
    # Common operations
    logging.basicConfig(level=options.pop("verbosity"))

    # Register new model version on MLFlow
    logger.info("Registering run_id %s as model %s", run_id, model_name)
    model_uri = f"runs:/{run_id}/model"
    logger.debug("Model URI: %s", model_uri)
    mlflow.register_model(f"runs:/{run_id}/model", model_name)

    # End of program
    logger.info("End of MNIST model training script")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
