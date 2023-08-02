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

The project structure is based on the Drivendata cookiecutter data science:
http://drivendata.github.io/cookiecutter-data-science/

To install the project you have to execute:

```bash
$ pip install -e .
```

An then all the scripts should be available inside python as modules.

To generate models, use the scripts inside the `deepaas_full.models` module.

```bash
$ python -m deepaas_full.models.mnist_convolution -n model_1 --learning_rate 1e-4
```

New modules are stored using MLflow, by default in the `models` folder. However
you can configure MLflow environment variables to use a remote server. More
details at https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded.

This example contains already generated models and autoencoders in the
`models` folder. You can use them to test the visualization scripts. To access
the MLflow UI, you need to execute:

```bash
$ mlflow ui --backend-store-uri models
```

Once a model is created, you will be able to select it on the scripts using the
`model_name` argument together with the `--version` option.

In order to train models, you need to generate datasets from MNIST data. This
example uses [DVC](https://dvc.org/) to manage the data. All data are stored
in the `data` folder. To download the raw data, you need to execute:

```bash
$ dvc pull
```

To generate the datasets, you need to execute the scripts located in the
`deepaas_full.data` module. For example:

```bash
$ python -m deepaas_full.data.make_convolution -o file_name images.gz labels.gz
```

Some scripts might require the use of an autoencoder model to generate the
encoded data. You can use the autoencoder models provided in `models` folder.

Once you have the datasets, you can train the models. You can use the script
provided in as `deepaas_full.models.train_model` module. For example:

```bash
$ python -m deepaas_full.models.train_model model_1 --version=1 dataset.npz
```

Once a model is trained, you will receive a run_id from MLflow. You can use
this run_id to visualize the find the training in MLflow UI with the metrics
and parameters used.

If your training is successful, you can register the run_id model using the
script `register_model` with passing the run_id and the model_name you want to
add the new version. For example:

```bash
$ python -m deepaas_full.models.register_model a00aa..000aaa model_1
```

This creates a new version of the model for the selected name. Note that some
scripts use `Production` model version as default. You assign a model version
to `Staging` or `Production` by using the `--staging` or `--production` option
in the script `deepaas_full.models.stage_model`. For example:

```bash
$ python -m deepaas_full.models.stage_model --production model_1 --version=1
```

Similarly you can provide `--staging` to set a model to staging. By default,
when no version is indicated, the selected stage is applied to the latest

> You can use the web interface (mlflow ui) to perform this step manually.

Once you have generated and trained your model, you can use the scripts
provided in `deepaas_full.visualization` to visualize the results. For example:

```bash
$ python -m deepaas_full.visualization.plot_predictions model_1 images.gz labels.gz
```
