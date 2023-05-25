"""This model example builds a simple feedforward neural network (with one
hidden layer) from scratch with NumPy to recognize handwritten digit images.

The deep learning model — one of the most basic artificial neural networks
that resembles the original multi-layer perceptron — will learn to classify
digits from 0 to 9 from the MNIST dataset.

The dataset contains 60,000 training and 10,000 test images and corresponding
labels. Each training and test image is of size 784 (or 28x28 pixels), this
is the input for the neural network.

Based on the image inputs and their labels (supervised learning), the neural
network is trained to learn their features using forward propagation and
backpropagation (reverse-mode differentiation). The final output of the
network is a vector of 10 scores — one for each handwritten digit image. You
will also evaluate how good the model is at classifying the images on the
test set.

Based on "Deep learning on MNIST" at https://github.com/numpy/numpy-tutorials
and "Tensorflow tutorials" https://www.tensorflow.org/tutorials/keras.
"""
import time

import tensorflow as tf
from keras import callbacks, layers

from deepaas_full import config, utils


def create_model(hidden_size=512, dropout_factor=0.2):
    """Creates a new MNIST model ready for training. The model is composed
    by 3 layers (input, hidden, output) with dropout after hidden layer. It
    uses a `relu` activation function on the hidden layer.

    Keyword Arguments:
        hidden_size -- Number of nodes for hidden layer (default: {512})
        dropout_factor -- Dropout after hidden layer (default: {0.2})

    Returns:
        Tensorflow MNIST model ready for training.
    """
    model = tf.keras.Sequential(
        [
            layers.Dense(hidden_size, "relu", input_shape=config.INPUT_SHAPE),
            layers.Dropout(dropout_factor),
            layers.Dense(config.LABEL_DIMENSIONS),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives(),
        ],
    )
    return model


def predict(model, input_data, **options):
    """Performs predictions on data using a MNIST model.

    Arguments:
        model -- Tensorflow/Keras model to use for predictions.
        input_data -- GZip file with images equivalent to MNIST data.
        options -- See tensorflow/keras predict documentation.

    Returns:
        Tuple with prediction values and None as callbacks history.
    """
    predict_data = utils.Dataset(input_data).data
    options["callbacks"] = generate_callbacks()
    predictions = model.predict(*predict_data, verbose="auto", **options)
    return predictions, None


def training(model, input_data, target_data, **options):
    """Performs training on a model from raw MNIST input and target data.

    Arguments:
        model -- Tensorflow/Keras model to train with data.
        input_data -- GZip file with training images for MNIST data.
        target_data -- GZip file with training labels for MNIST data.
        options -- See tensorflow/keras fit documentation.

    Returns:
        Tuple with callbacks history and None as prediction values.
    """
    train_data = utils.Training(input_data, target_data).data
    options["callbacks"] = generate_callbacks()
    _callbacks = model.fit(*train_data, verbose="auto", **options)
    return None, _callbacks.history


def generate_callbacks():
    """Generator for for training callbacks. It includes the generation
    of model checkpoints to be saved at the configured models path.

    Returns:
        Tensorflow training callbacks list.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = config.MODELS_PATH / f"{timestamp}.cp.ckpt"
    return [callbacks.ModelCheckpoint(checkpoint_path, verbose=1)]
