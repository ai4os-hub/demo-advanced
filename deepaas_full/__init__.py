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

import numpy as np

from deepaas_full import config, utils, core

from dataclasses import dataclass
import numpy.typing as npt


rng = np.random.default_rng(int(config.RAND_SEED))


class ModelMNIST(core.BaseModel):
    """Generates a MNIST Neural Network model capable of predicting a number
    between 0 and 9 from hand writings.

    Returns:
        Instance of MNIST model based on neural networks.
    """

    num_labels = int(config.LABEL_DIMENSIONS)
    image_size = int(config.IMAGE_SIZE)

    def __init__(self, hidden_size=100, **kwds) -> None:
        super().__init__(**kwds)
        self.w[1] = 0.2 * rng.random((self.image_pixels, hidden_size)) - 0.1
        self.w[2] = 0.2 * rng.random((hidden_size, self.num_labels)) - 0.1

    @property
    def image_pixels(self):
        return self.image_size**2

    @classmethod
    def preprocess_data(cls, datas, func=lambda x: x / 255):
        return func(datas)

    @classmethod
    def preprocess_label(cls, labels, func=utils.one_hot_encoding):
        return func(labels, dimension=cls.num_labels)


@dataclass
class ForwardData:
    dropout_mask: npt.ArrayLike
    layer_0: npt.ArrayLike
    layer_1: npt.ArrayLike
    layer_2: npt.ArrayLike


def forward_propagation(model: ModelMNIST, input_data):
    layer_1 = np.dot(input_data, model.w[1])
    layer_1 = utils.relu(layer_1)
    dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
    layer_1 *= dropout_mask * 2
    layer_2 = np.dot(layer_1, model.w[2])
    return ForwardData(dropout_mask, input_data, layer_1, layer_2)


@dataclass
class BackpropData:
    delta_1: npt.ArrayLike
    delta_2: npt.ArrayLike


def back_propagation(model: ModelMNIST, pdata: ForwardData, label):
    delta_2 = label - pdata.layer_2
    delta_1 = np.dot(model.w[2], delta_2)
    delta_1 *= utils.relu2deriv(pdata.layer_1)
    delta_1 *= pdata.dropout_mask
    return BackpropData(delta_1, delta_2)


def training_step(model: ModelMNIST, learning_rate, loss=0.0, acc=0):
    for data, label in model.training:
        fdata = forward_propagation(model, data)
        bdata = back_propagation(model, fdata, label)
        model.w[1] += learning_rate * np.outer(fdata.layer_0, bdata.delta_1)
        model.w[2] += learning_rate * np.outer(fdata.layer_1, bdata.delta_2)
        loss += np.sum((label - fdata.layer_2) ** 2)
        acc += int(np.argmax(fdata.layer_2) == np.argmax(label))
    return loss, acc


def evaluation_step(model: ModelMNIST):
    result = utils.relu(model.x_test @ model.w[1]) @ model.w[2]
    loss = np.sum((model.y_test - result) ** 2)
    acc = np.sum(np.argmax(result, axis=1) == np.argmax(model.y_test, axis=1))
    return loss, acc


def train(model: ModelMNIST, epochs, learning_rate=0.005):
    stats = {k: core.ExecutionStats() for k in ["train", "test"]}
    for _ in range(epochs):
        loss, acc = training_step(model, learning_rate)
        stats["train"].append(loss, acc, model.training_len)
        loss, acc = evaluation_step(model)
        stats["test"].append(loss, acc, model.testing_len)
    return stats


def predict(model: ModelMNIST, input_data):
    return [forward_propagation(model, data).layer_2 for data in input_data]
