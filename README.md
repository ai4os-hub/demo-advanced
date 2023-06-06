# DEEPaaS Full Template

> TODO: Complete build status
> [![Build Status]()]()

## Download and Installation

To launch it, first download the repository and install the package using pip.

```bash
git clone https://github.com/.../{repostory}  # Download repository
cd {repository}  # Navigate inside repository project
pip install -e .  # Install repository project
```

> Use editable mode `pip install -e .` when you are testing your patch to a
> package through another project.

## Data Version Control

As second step, you need to download your data and datasets. This project uses
the Data Version Control [DVC](https://dvc.org/) library to organize and
version all the dataset

For tutorial about how to use `dvc` follow the following guides
https://dvc.org/doc/start/data-management

```bash
dvc remote modify --local deep-cloud user {your-dvc-remote-user}
dvc remote modify --local deep-cloud password {your-dvc-remote-password}
dvc pull  # Download data from your dvc remote storage
```

> If your provider does not provide webdav, check the alternatives offered at
> [DVC Remotes](https://dvc.org/doc/user-guide/data-management/remote-storage)
> and your storage provider.

Make sure your MNIST data repository provides a list of `npz` datasets at the
folder `./data` or your folder configured as _DATA_PATH_ in your environment.
The API metadata will provide all the names of the datasets available in such
folder.

> Note the model functions expect `npz` datasets with the keys `images` and
> `labels`.

## FLFlow Experiments and Models Repository

Next step is to configure your experiments, training and models repository.
This example uses [MLFlow](https://mlflow.org/) in order to track and store
your models. In order to work correctly, you can configure the following
environment variables:

- _MLFLOW_TRACKING_URI_ pointing to the MLflow server to use for storing models.
- _MLFLOW_TRACKING_USERNAME_ username to use with HTTP Basic authentication on MLflow.
- _MLFLOW_TRACKING_PASSWORD_ password to use with HTTP Basic authentication on MLflow.
- _MLFLOW_EXPERIMENT_NAME_ experiment identification to store trainings on MLflow.
- _MLFLOW_EXPERIMENT_ID_ alternative to _MLFLOW_EXPERIMENT_NAME_ by on MLflow.

You can use this by creating an `.env` loaded by your script or directly from
the command line:

```bash
export MLFLOW_TRACKING_USERNAME={your-mflow-user}
export MLFLOW_TRACKING_PASSWORD={your-mflow-password}
export MLFLOW_TRACKING_URI=https://{your-mlflow-address}:{mlflow-port}
export MLFLOW_EXPERIMENT_ID={your-mflow-experiment-id}
```

> Username and password are only required on MLFlow deployment protected by user and password.

## Configure and run DEEPaaS

To configure DEEPaaS functionalities, create a copy from `deepaas.conf.sample`,
customize it with your preferred values and pass it to the run call as
`deepaas-run --config-file deepaas.conf`.
More information about how to configure DEEPaaS can be found a the
[official documentation](https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/stable/install/configuration/index.html)

```bash
deepaas-run --config-file=deepaas.conf  # You deploy using deepaas
```

> Note that some ports like `80` require administrative privileges. Use `sudo`
> and ensure that `deepaas-run` script is available in the root `PATH`, for
> example, if you are using conda
> `sudo env PATH=$PATH:/{path-DEEPaaS-bin} deepaas-run --config-file=deepaas.conf`
> together with the script to load your environment variables.

## Model and API additional configuration

Additionally you can configure the following environment variables for customization:

API configuration environment variables:

- _DATA_PATH_ path pointing to the training datasets, default `./data/raw`.
- _MODEL_NAME_ package name used to provide API metadata, default `deepaas_full`.

Model configuration environment variables:

- _LABEL_DIMENSIONS_ dimensions the labels are hot encoded, default `10`.
- _IMAGE_SIZE_ vertical and horizontal pixels per image, default `28`.

## Testing

Testing process is automated by tox. Check the environments configured to be
tested by `tox --listenvs`. If you are missing one of the python environments
configured to be tested and you are using conda for managing your virtual
environments, consider using `tox-conda` to automatically manage all python
installation on your testing virtual environment.

Tests are implemented following [pytest](https://docs.pytest.org) framework.
Fixtures and parametrization are placed inside `conftest.py` files meanwhile
assertions are located on `test_{}.py` files.

Tests are performed by a remote model named `deepaas-full-testing` using its
version `1`. In order to pass all tests, you need to provide this model on
your MLFlow model repository and configure the environment variables to access
it. Experiments can be tracked if you set a _MLFLOW_EXPERIMENT_NAME_ or
_MLFLOW_EXPERIMENT_id_ however, leaving empty those variables will avoid the
generation of experiment tracking on the MLFlow server.

> If tests fail with `Registered Model with name=deepaas-full-testing not found`
> but you are sure that the model `deepaas-full-testing` exists in your MLFlow
> repository, ensure your MLFlow environment is accessible at test run time.

## Project structure

After downloading and configuring the your project instance, your project
folder should look approximately as follows:

```
├── .dvc                    <- Folder with Data Version Control configuration
├── .dvcignore              <- Untracked files that dvc should ignore
├── .env                    <- Environment configuration file
├── .env.sample             <- Environment configuration sample
├── .git                    <- Folder with Code Version Control data (git)
├── .gitignore              <- Untracked files that git should ignore
├── .pytest_cache           <- Cache generated by pytest framework
├── .tox                    <- Environments generated by tox automated testing
├── .vscode                 <- VSCode configuration (debug, formatting, etc.)
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- Project and model license file
├── README.md               <- The top-level README for using this project
├── VERSION                 <- File indicating the API/Model version
├── api                     <- Project package for DEEPaaS API
├── data                    <- Folder with model datasets and raw data
├── deepaas.conf            <- DEEPaaS configuration file
├── deepaas.conf.sample     <- DEEPaaS configuration sample
├── deepaas_full            <- Package folder containing the model code
├── deepaas_full.egg-info   <- Pip build for package installation
├── dvc.lock                <- Data record and output state tracking (dvc)
├── dvc.yaml                <- Configuration and stages for dvc
├── pyproject.toml          <- Makes project installable (pip install -e .)
├── htmlcov                 <- Report from tox qc.cov environment
├── requirements-test.txt   <- Requirements file for testing the service
├── requirements.txt        <- Requirements file for running the service
├── tests                   <- Folder containing tests for the API methods
└── tox.ini                 <- Automated testing configuration file for tox
```
