import gzip

import numpy as np


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
