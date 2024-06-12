# AI4OS Demo Advanced

> TODO: Add badges for CI/CD, coverage, license, etc. from AI4OS Jenkins.

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/demo-advanced/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/demo-advanced/job/main/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)

## Download, Installation and Requirements

To launch it, first download the repository and install the package using pip.

```bash
git clone git@github.com:ai4os-hub/demo-advanced.git  # Download repository
cd {repository}  # Navigate inside repository project
pip install -U pip  # Upgrade pip to avoid errors
pip install -e .  # Install repository project
```

> Use editable mode `pip install -e .` when you are want python to access your
> your code at the repository and not a copy at `.../site-packages`. Useful to
> test your package changes without reinstalling.
> Old versions of pip require `setup.py` for installation, upgrade `pip` as
> indicated on the instructions to solve this error.

## Data implementation

As second step, you need to download MNIST dataset and place it inside the
`data/raw` with the following structure:

```bash
data/raw
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte.gz
└── train-labels-idx1-ubyte.gz
```

You can change the data folder by setting the environment variable
_DEMO_ADVANCED_DATA_URI_ to the path of your data folder.
Once you have downloaded the data, you can generate processed dataset by
running the following scripts:

- `python -m demo_advanced.data.make_convolution` for convolution training data.
- `python -m demo_advanced.data.make_autoencoder` for autoencoders training data.
- `python -m demo_advanced.data.make_encoded` for encoded training data.

One example of the expected data structure is the following:

```bash
data/processed
├── t10k-convolution.npz
├── t10k-encoded.npz
├── t60k-autoencoder.npz
├── t60k-convolution.npz
└── t60k-encoded.npz
```

> Note model function `demo_advanced.train` expects `npz` files with keys
> `images` and `labels`.

## Model implementation and storage

This example uses different models to train and predict MNIST dataset.
By default, models are stored in the local `models` folder. You can change
the model folder by setting the environment variable _DEMO_ADVANCED_MODELS_URI_.
To generate the models, you can run the following scripts:

- `python -m demo_advanced.models.make_autoencoder` for autoencoder model.
- `python -m demo_advanced.models.make_convolution` for convolution model.
- `python -m demo_advanced.models.make_dense2ly` for 2 layers full connected model.

## Configure and run DEEPaaS

To configure DEEPaaS functionalities, create a copy from `deepaas.conf.sample`,
customize it with your preferred values and pass it to the run call as
`deepaas-run --config-file deepaas.conf`.
More information about how to configure DEEPaaS can be found a the
[official documentation](https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/install/configuration/index.html)

```bash
deepaas-run --config-file=deepaas.conf  # You deploy using deepaas
```

> Note that some ports like `80` or `443` require administrative privileges.
> Use `sudo` and ensure that `deepaas-run` script is available in root `PATH`,
> for example, if you are using conda
> `sudo env PATH=$PATH:/{path-DEEPaaS-bin} deepaas-run --config-file=deepaas.conf`
> together with your script or method to load your environment variables.

## Model and API additional configuration

Additionally you can configure the following environment variables for customization:

API configuration environment variables:

- _DEMO_ADVANCED_MODELS_URI_ pointing to the models folder, default `./models`.
- _DEMO_ADVANCED_DATA_URI_ pointing to the training datasets, default `./data`.

Model data configuration environment variables:

- _DEMO_ADVANCED_LABEL_DIMENSIONS_ dimensions the labels are hot encoded, default `10`.
- _DEMO_ADVANCED_IMAGE_SIZE_ vertical and horizontal pixels per image, default `28`.

## Testing

Testing process is automated by tox library. You can check the environments
configured to be tested by running `tox --listenvs`. If you are missing one
of the python environments configured to be tested (e.g. py310, py39) and
you are using `conda` for managing your virtual environments, consider using
`tox-conda` to automatically manage all python installation on your testing
virtual environment.

```bash
tox -e {environment}  # Run tests on a specific environment
```

Tests are implemented following [pytest](https://docs.pytest.org) framework.
Fixtures and parametrization are placed inside `conftest.py` files meanwhile
assertion tests are located on `test_*.py` files.

```bash
python -m pytest tests
```

This example provides an example of minimalistic dataset and model to run
all testing functionalities. As the dataset and model are minimalistic, it is
supposed to be fast to run all tests. If you want to test your own dataset
see how to configure the environment variables for testing at pyproject.toml
using [pytest-env](https://pypi.org/project/pytest-env/).

## Project structure

After downloading and configuring the your project instance, your project
folder should look approximately as follows:

```
├── .env                    <- Environment configuration file
├── .env.sample             <- Environment configuration sample
├── .git                    <- Folder with Code Version Control data (git)
├── .gitignore              <- Untracked files that git should ignore
├── .pytest_cache           <- Cache generated by pytest framework
├── .sqa                    <- AI4OS Software Quality Assurance (SQA) folder
├── .tox                    <- Environments generated by tox automated testing
├── .vscode                 <- VSCode configuration (debug, formatting, etc.)
├── Dockerfile              <- Instructions to build service containers
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- Project and model license file
├── README.md               <- The top-level README for using this project
├── VERSION                 <- File indicating the API/Model version
├── api                     <- Project package for DEEPaaS API
├── data                    <- Folder with model datasets and raw data
├── deepaas.conf            <- DEEPaaS configuration file
├── deepaas.conf.sample     <- DEEPaaS configuration sample
├── demo_advanced           <- Package folder containing the model code
├── docs                    <- Folder with documentation files
├── htmlcov                 <- Report from tox qc.cov environment
├── metadata.json           <- Defines information to the [DEEP Open Catalog](https://marketplace.deep-hybrid-datacloud.eu)
├── models                  <- Folder with local MLFlow models
├── pyproject.toml          <- Makes project installable (pip install -e .)
├── notebooks               <- Jupyter notebooks folder
├── references              <- Data dictionaries, manuals, and explanatory materials
├── requirements-dev.txt    <- Requirements file with development utilities
├── requirements-test.txt   <- Requirements file for testing the service
├── requirements.txt        <- Requirements file for running the service
├── tests                   <- Folder containing tests for the API methods
└── tox.ini                 <- Automated testing configuration file for tox
```
