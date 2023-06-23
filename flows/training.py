import time
from uuid import UUID

import httpx
import prefect


@prefect.flow
def train_model(
    model_uri: str = "models:/borja-MNIST/1",
    dataset: str = "t10k-dataset.npz",
    max_retries: int = 10,
):
    uuid = post_training(model_uri, dataset)
    for retry in range(max_retries):
        training = get_training(uuid)
        if training["status"] == "running":
            time.sleep(10)
        else:
            break

    if retry >= 10:
        raise TimeoutError("Training timeout")


@prefect.task
def post_training(model_uri: str, dataset: str) -> UUID:
    logger = prefect.get_run_logger()
    response = httpx.post(
        "http://localhost:5000/v2/models/deepaas_full/train/",
        params={"model_uri": model_uri, "dataset": dataset},
    )
    response.raise_for_status()
    logger.info("Response: %s", response.json())
    return response.json()["uuid"]


@prefect.task
def get_training(uuid: UUID):
    logger = prefect.get_run_logger()
    response = httpx.get(
        f"http://localhost:5000/v2/models/deepaas_full/train/{uuid}",
    )
    response.raise_for_status()
    logger.info("Response: %s", response.json())
    return response.json()


if __name__ == "__main__":
    train_model()
