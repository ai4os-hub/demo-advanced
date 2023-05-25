"""Utils module example for MNIST DEEPaaS Full demo.
"""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import gzip

import numpy as np

from deepaas_full import config


def one_hot_encoding(labels, dimension=10):
    """Define a one-hot variable for an all-zero vector with 10 dimensions.
    (number labels from 0 to 9; or N dimension)
    """
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float64)


class Dataset(object):
    def __init__(self, images_path):
        with gzip.open(images_path, "rb") as file:
            raw_x = np.frombuffer(file.read(), np.uint8, offset=16)
            raw_x = raw_x.reshape(-1, 28 * 28)
        self.images = Training.preprocess_images(raw_x)

    @classmethod
    def preprocess_images(cls, data, func=lambda x: x / 255):
        return func(data)

    @property
    def data(self):
        return (self.images,)


class Training(Dataset):
    def __init__(self, images_path, labels_path):
        super().__init__(images_path)
        with gzip.open(labels_path, "rb") as file:
            raw_y = np.frombuffer(file.read(), np.uint8, offset=8)
        self.labels = Training.preprocess_labels(raw_y)

    @classmethod
    def preprocess_labels(cls, labels, func=one_hot_encoding):
        return func(labels, dimension=config.LABEL_DIMENSIONS)

    @property
    def data(self):
        return (self.images, self.labels)
