"""Utils module example for MNIST DEEPaaS Full demo.
"""
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

import gzip

import tensorflow as tf
import numpy as np

from deepaas_full import config


class Dataset(object):
    def __init__(self, images_path):
        with gzip.open(images_path, "rb") as file:
            raw_x = np.frombuffer(file.read(), np.uint8, offset=16)
            raw_x = raw_x.reshape(-1, config.IMAGE_SIZE, config.IMAGE_SIZE)
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
    def preprocess_labels(cls, labels, func=tf.keras.utils.to_categorical):
        return func(labels, config.LABEL_DIMENSIONS)

    @property
    def data(self):
        return (self.images, self.labels)
