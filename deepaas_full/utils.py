import gzip

import numpy as np

from deepaas_full import config


def raw_images(file_path):
    with gzip.open(file_path, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        return data.reshape(-1, 28 * 28)


def raw_labels(file_path):
    with gzip.open(file_path, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=8)
        return data


def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zero vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == np.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(np.float64)


class Training(object):
    def __init__(self, input_data, target_data):
        raw_x, raw_y = raw_images(input_data), raw_labels(target_data)
        self.images = Training.preprocess_images(raw_x)
        self.labels = Training.preprocess_labels(raw_y)

    @classmethod
    def preprocess_images(cls, data, func=lambda x: x / 255):
        return func(data)

    @classmethod
    def preprocess_labels(cls, labels, func=one_hot_encoding):
        return func(labels, dimension=config.LABEL_DIMENSIONS)

    @property
    def data(self):
        return (self.images, self.labels)
