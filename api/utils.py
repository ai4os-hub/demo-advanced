"""Utilities module for API endpoints and methods.
"""
import logging
import subprocess  # nosec B404
import sys
from pathlib import Path

import mlflow

from . import config

tracking_uri = str(config.MODELS_PATH)  # pylint: disable=invalid-name
mlflow_client = mlflow.MlflowClient(tracking_uri)
logger = logging.getLogger(__name__)


def ls_models():
    """Utility to return a list of models available in the MLFlow connected
    models registry.

    Returns:
        A list of RegisteredModel from mlflow.
    """
    models = mlflow_client.search_registered_models()
    return {x.name: x.description for x in models}


def ls_datasets():
    """Utility to return a list of datasets available in `data` folder.

    Returns:
        A list of strings in the format {id}-{type}.npz.
    """
    processed_path = Path(config.DATA_PATH) / "processed"
    logger.debug("Scanning at: %s", processed_path)
    dirscan = (x.name for x in processed_path.glob("*.npz"))
    return sorted(dirscan)


def copy_remote(frompath, topath, timeout=600):
    """Copies remote (e.g. NextCloud) folder in your local deployment or
    vice versa for example:
        - `copy_remote('rshare:/data/images', '/srv/myapp/data/images')`

    Subprocess might introduce vulnerabilities in the code. In order to reduce
    security vulnerabilities, it is recommended to be used with shell=False.
    https://security.openstack.org/guidelines/dg_use-subprocess-securely.html#correct

    Arguments:
        frompath -- Source folder to be copied.
        topath -- Destination folder.
        timeout -- Timeout in seconds for the copy command.

    Returns:
        A tuple with stdout and stderr from the command.
    """
    with subprocess.Popen(  # nosec B603
        args=["rclone", "copy", f"{frompath}", f"{topath}"],
        stdout=subprocess.PIPE,  # Capture stdout
        stderr=subprocess.PIPE,  # Capture stderr
        shell=False,  # No shell for security reasons
        text=True,  # Return strings rather than bytes
    ) as process:
        try:
            outs, errs = process.communicate(None, timeout)
        except subprocess.TimeoutExpired:
            logger.error("Timeout when copying from/to remote directory.")
            process.kill()
            outs, errs = process.communicate()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error copying from/to remote directory\n %s", exc)
            process.kill()
            outs, errs = process.communicate()
    return outs, errs


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
