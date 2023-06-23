"""Training workflow for MNIST model.
"""
import argparse
import sys

import prefect
from prefect.task_runners import SequentialTaskRunner

import api

# Script arguments definition ---------------------------------------
parser = argparse.ArgumentParser(
    prog="PROG",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="See '<command> --help' to read about a specific sub-command.",
)
parser.add_argument(
    "model_uri",
    help="Model URI from MLFlow to use for training.",
    type=str,
)
parser.add_argument(
    "dataset",
    help="Dataset to use for training.",
    type=str,
)
# -------------------------------------------------------------------


@prefect.flow(
    name="MNIST-Training-Flow",
    description="Flow to train a MNIST model.",
    task_runner=SequentialTaskRunner(),
)
def main(model_uri, dataset, **options):
    train(model_uri, dataset)


@prefect.task
def train(model_uri, dataset, **options):
    logger = prefect.get_run_logger()
    result = api.train(model_uri, dataset, **options)
    logger.debug("Response: %s", result)
    return result


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
    sys.exit(0)  # Shell return 0 == success
