"""This model example to build feedforward neural network models from scratch
with Tensorflow and keras to recognize handwritten digit images.

The deep learning model — one of the most basic artificial neural networks
that resembles the original multi-layer perceptron — will learn to classify
digits from 0 to 9 from the MNIST dataset.

Based on the image inputs and their labels (supervised learning), the neural
network is trained to learn their features using forward propagation and
backpropagation (reverse-mode differentiation). The final output of the
network is a vector of 10 scores — one for each handwritten digit image. You
will also evaluate how good the model is at classifying the images on the
test set.

Based on "Deep learning on MNIST" at https://github.com/numpy/numpy-tutorials
and "Tensorflow tutorials" https://www.tensorflow.org/tutorials/keras.
"""
import logging

import mlflow
import numpy as np

# Configuration is mandatory to execute when running package scripts
from deepaas_full import config

logger = logging.getLogger(__name__)


def predict(input_file, model_name, version="production", **options):
    """Performs predictions on data using a MNIST model.

    Arguments:
        input_file -- NPY file with images equivalent to MNIST data.
        model_name -- MLFlow model name to use for predictions.
        version -- MLFLow model version to use for predictions.
        options -- See tensorflow/keras predict documentation.

    Returns:
        Return value from tf/keras model predict.
    """
    logger.debug("Using model %s for predictions", model_name)
    model_uri = f"models:/{model_name}/{version}"
    logger.debug("Loading model from uri: %s", model_uri)
    model = mlflow.tensorflow.load_model(model_uri)
    logger.debug("Loading data from input_file: %s", input_file)
    input_data = np.load(input_file)
    logger.debug("Predict with options: %s", options)
    return model.predict(input_data, verbose="auto", **options)


def training(input_file, model_name, version="production", **options):
    """Performs training on a model from raw MNIST input and target data.

    Arguments:
        input_file -- NPZ file with training images and labels.
        model_name -- MLFlow model name to use for predictions.
        version -- MLFLow model version to use for predictions.
        options -- See tensorflow/keras fit documentation.

    Returns:
        Return value from tf/keras model fit.
    """
    logger.debug("Using model %s for training", model_name)
    model_uri = f"models:/{model_name}/{version}"
    logger.debug("Loading model from uri: %s", model_uri)
    model = mlflow.tensorflow.load_model(model_uri)
    logger.debug("Loading data from input_file: %s", input_file)
    with np.load(input_file) as input_data:
        train_data = input_data["x_train"], input_data["y_train"]
    logger.debug("Training with options: %s", options)
    with mlflow.start_run(nested=False) as run:
        mlflow.tensorflow.autolog()
        model.fit(*train_data, verbose="auto", **options)
    return run.info.run_id
