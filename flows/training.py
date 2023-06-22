import httpx
from prefect import flow, get_run_logger


@flow
def train_model(
    model_uri: str = "models:/borja-MNIST/1",
    dataset: str = "t10k-dataset.npz",
):
    logger = get_run_logger()
    response = httpx.post(
        "http://localhost:5000/v2/models/deepaas_full/train/",
        params={"model_uri": model_uri, "dataset": dataset},
    )
    response.raise_for_status()
    logger.info("Response: %s", response.json())


if __name__ == "__main__":
    train_model()
