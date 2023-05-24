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

Based on "Deep learning on MNIST" at https://github.com/numpy/numpy-tutorials.
"""
import time

import tensorflow as tf
from keras import callbacks, layers, losses, metrics, optimizers

from deepaas_full import config, utils


def create_model(hidden_size=512, dropout_factor=0.2):
    model = tf.keras.Sequential(
        [
            layers.Dense(hidden_size, "relu", input_shape=config.INPUT_SHAPE),
            layers.Dropout(dropout_factor),
            layers.Dense(config.LABEL_DIMENSIONS),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )
    return model


def training(model, input_data, target_data, **options):
    # model = utils.load_model(model_name)
    train = utils.Training(input_data, target_data)
    options["callbacks"] = generate_callbacks()
    model.fit(train.data, verbose="auto", **options)
    return model.summary()


def generate_callbacks():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = config.MODELS_PATH / f"{timestamp}/cp.ckpt"
    return [callbacks.ModelCheckpoint(checkpoint_path, verbose=1)]
