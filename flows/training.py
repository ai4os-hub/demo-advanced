"""Training workflow for MNIST model.
"""
import prefect
from prefect.task_runners import SequentialTaskRunner

import api


@prefect.flow(
    name="MNIST-Training-Flow",
    description="Flow to train a MNIST model.",
    task_runner=SequentialTaskRunner(),
)
def main(model_uri, dataset):
    train(model_uri, dataset)


@prefect.task
def train(model_uri, dataset, **options):
    logger = prefect.get_run_logger()
    result = api.train(model_uri, dataset, **options)
    logger.debug("Response: %s", result)
    return result
