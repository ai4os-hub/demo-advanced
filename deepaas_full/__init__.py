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
num_labels = int(config.LABEL_DIMENSIONS)
image_size = int(config.IMAGE_SIZE)
image_pixels = image_size**2


@dataclass
class ForwardData:
    dropout_mask: npt.ArrayLike
    layer_0: npt.ArrayLike
    layer_1: npt.ArrayLike
    layer_2: npt.ArrayLike


@dataclass
class BackpropData:
    delta_1: npt.ArrayLike
    delta_2: npt.ArrayLike


class ModelMNIST(core.BaseModel):
    """Generates a MNIST Neural Network model capable of predicting a number
    between 0 and 9 from hand writings.

    Returns:
        Instance of MNIST model based on neural networks.
    """

    def __init__(self, hidden_size=100, **kwds) -> None:
        super().__init__(**kwds)
        self.w[1] = 0.2 * rng.random((image_pixels, hidden_size)) - 0.1
        self.w[2] = 0.2 * rng.random((hidden_size, num_labels)) - 0.1

    def predict(self, input_data):
        return [self.forward_propagation(data).layer_2 for data in input_data]

    def forward_propagation(self, input_data):
        layer_1 = np.dot(input_data, self.w[1])
        layer_1 = utils.relu(layer_1)
        dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = np.dot(layer_1, self.w[2])
        return ForwardData(dropout_mask, input_data, layer_1, layer_2)

    def back_propagation(self, pdata: ForwardData, label):
        delta_2 = label - pdata.layer_2
        delta_1 = np.dot(self.w[2], delta_2)
        delta_1 *= utils.relu2deriv(pdata.layer_1)
        delta_1 *= pdata.dropout_mask
        return BackpropData(delta_1, delta_2)


class TrainingMNIST(core.Training):
    """Generates a MNIST training instance capable of training a ModelMNIST
    to improve predictions.

    Returns:
        Instance of MNIST training based on neural networks.
    """

    @classmethod
    def preprocess_data(cls, datas, func=lambda x: x / 255):
        return func(datas)

    @classmethod
    def preprocess_label(cls, labels, func=utils.one_hot_encoding):
        return func(labels, dimension=num_labels)

    def evaluation_step(self, model: ModelMNIST):
        results = utils.relu(self.x_test @ model.w[1]) @ model.w[2]
        hits = np.argmax(results, axis=1) == np.argmax(self.y_test, axis=1)
        loss = np.sum((self.y_test - results) ** 2)
        return loss, np.sum(hits)

    def training_step(self, model: ModelMNIST, learning_rate):
        loss, hits = 0.0, 0
        for data, label in self.training:
            fd = model.forward_propagation(data)
            bd = model.back_propagation(fd, label)
            model.w[1] += learning_rate * np.outer(fd.layer_0, bd.delta_1)
            model.w[2] += learning_rate * np.outer(fd.layer_1, bd.delta_2)
            loss += np.sum((label - fd.layer_2) ** 2)
            hits += int(np.argmax(fd.layer_2) == np.argmax(label))
        return loss, hits
