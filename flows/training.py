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
def main(mlflow_uri, model_uri, dataset, experiment_id=None, **options):
    logger = prefect.get_run_logger()
    result = train(mlflow_uri, model_uri, dataset, experiment_id, **options)
    logger.debug("Training result: %s", result)
    return result


@prefect.task
def train(mlflow_uri, model_uri, dataset, experiment_id, **options):
    return api.train(mlflow_uri, model_uri, dataset, experiment_id, **options)
