# Python MNIST sample project for Model

Simple project sample to build feedforward neural network models from scratch
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

## Generate dataset
```bash
$ python -m deepaas_full.data.make_convolution -o file_name images.gz labels.gz
```

## Create models
```bash
$ python -m deepaas_full.models.mnist_convolution -n model_1 --learning_rate 1e-4
```

## Set model to production
```bash
$ python -m deepaas_full.models.set_stage --stage=Production model_1 --version=1
```

## Visualize data
```bash
$ python -m deepaas_full.visualization.plot_samples -i=5
```
