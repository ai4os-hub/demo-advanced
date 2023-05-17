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

Based on "Deep learning on MINST" at https://github.com/numpy/numpy-tutorials.
"""
import pathlib
import abc

from . import config, utils


class BaseModel(abc.ABC):
    """Base class to with abstract methods and generic configuration to define
    specific data models classes based on numpy and neural networks.
    """

    data_path = pathlib.Path(config.DATA_PATH)
    label_dimensions = int(config.LABEL_DIMENSIONS)

    @classmethod
    @abc.abstractmethod
    def preprocess_data(cls, datas, func=None):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def preprocess_label(cls, labels, func=None):
        raise NotImplementedError


class HasTrain(BaseModel, abc.ABC):
    """Extension for BaseModel that includes trining images and labels.

    Arguments:
        test_images -- FIle name for raw test images.
        test_labels -- File name for raw test labels.
    """

    def __init__(self, **kwds) -> None:
        kwds = self.load_training(**kwds)

    def load_training(self, training_images, training_labels, **kwds):
        data_path = pathlib.Path(self.data_path) / "raw"
        self.x_train = utils.raw_images(data_path / training_images)
        self.y_train = utils.raw_labels(data_path / training_labels)
        return kwds

    @property
    def x_train(self):
        """The x_train property."""
        return self._x_train

    @x_train.setter
    def x_train(self, value):
        self._x_train = self.preprocess_data(value)

    @property
    def y_train(self):
        """The y_train property."""
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = self.preprocess_label(value)


class HasTest(BaseModel, abc.ABC):
    """Extension for BaseModel that includes test images and labels.

    Arguments:
        test_images -- FIle name for raw test images.
        test_labels -- File name for raw test labels.
    """

    def __init__(self, **kwds) -> None:
        kwds = self.load_test(**kwds)

    def load_test(self, test_images, test_labels, **kwds):
        data_path = pathlib.Path(self.data_path) / "raw"
        self.x_test = utils.raw_images(data_path / test_images)
        self.y_test = utils.raw_labels(data_path / test_labels)
        return kwds

    @property
    def x_test(self):
        """The x_test property."""
        return self._x_test

    @x_test.setter
    def x_test(self, value):
        self._x_test = self.preprocess_data(value)

    @property
    def y_test(self):
        """The y_test property."""
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = self.preprocess_label(value)


class ModelMINST(HasTest, HasTrain):
    """Generates a MINST Neural Network model capable of predicting a number
    between 0 and 9 from hand writings.

    Returns:
        Instance of MINST model based on neural networks.
    """

    def __init__(self, **kwds) -> None:
        HasTrain.__init__(self, **kwds)
        HasTest.__init__(self, **kwds)

    @classmethod
    def preprocess_data(cls, datas, func=lambda x: x / 255):
        return func(datas)

    @classmethod
    def preprocess_label(cls, labels, func=utils.one_hot_encoding):
        return func(labels, dimension=cls.label_dimensions)
