"""Script to set model stage version in mlflow.
"""
# pylint: disable=redefined-outer-name
import argparse
import logging
import sys

import mlflow

logger = logging.getLogger(__name__)
mlflow_client = mlflow.MlflowClient()


# Type validators ---------------------------------------------------
def version(string_value):
    """Validator converter for version values and/or digits."""
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
    *["model_name"],
    help="Model name to use for identification on mlflow registry.",
    type=str,
)
parser.add_argument(
    *["--version"],
    help="Model version to stage, latest version is collected if not defined",
    type=version,
)
parser.add_argument(
    *["--stage"],
    help="Stage version to set for model version (default: %(default)s).",
    type=str,
    choices=["Staging", "Production"],
    default="Production",
)


# Script command actions --------------------------------------------
def _run_command(model_name, version=None, **options):
    # Common operations
    logging.basicConfig(level=options.pop("verbosity"))
    logger.debug("Setting model %s version in production", model_name)

    # Collect latest model version if not defined
    if version is None:
        logger.info("Collecting latest model version")
        versions = mlflow_client.get_latest_versions(model_name)
        logger.debug("Versions: %s", versions)
        version = versions[0].version

    # Setting model version in staging or production
    logger.info("Setting version %s to stage %s", version, options["stage"])
    mlflow_client.transition_model_version_stage(
        model_name, version, options["stage"]
    )

    # End of program
    logger.info("End of setting model stage version")


# Main call ---------------------------------------------------------
if __name__ == "__main__":
    args = parser.parse_args()
    _run_command(**vars(args))
    sys.exit(0)  # Shell return 0 == success
