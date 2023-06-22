import uuid

import httpx
import prefect


@prefect.flow
def train_model(
    model_uri: str = "models:/borja-MNIST/1",
    dataset: str = "t10k-dataset.npz",
):
    train_request(model_uri, dataset)


@prefect.task
def train_request(model_uri: str, dataset: str) -> uuid.UUID:
    logger = prefect.get_run_logger()
    response = httpx.post(
        "http://localhost:5000/v2/models/deepaas_full/train/",
        params={"model_uri": model_uri, "dataset": dataset},
    )
    response.raise_for_status()
    logger.info("Response: %s", response.json())
    return response.json()["uuid"]


if __name__ == "__main__":
    train_model()
