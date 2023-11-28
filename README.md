![qc.sec](https://github.com/BorjaEst/demo-advanced-api/actions/workflows/qc-sec.yml/badge.svg)
![qc.sty](https://github.com/BorjaEst/demo-advanced-api/actions/workflows/qc-sty.yml/badge.svg)
![qc.uni](https://github.com/BorjaEst/demo-advanced-api/actions/workflows/qc-uni.yml/badge.svg)

# DEEPaaS Full Template

## Download, Installation and Requirements

To launch it, first download the repository and install the package using pip.

```bash
git clone git@git.scc.kit.edu:m-team/ai/demo-advanced-api.git  # Download repository
cd {repository}  # Navigate inside repository project
pip install -e .  # Install repository project
```

> Use editable mode `pip install -e .` when you are want python to access your
> your code at the repository and not a copy at `.../site-packages`. Useful to
> test your package changes without reinstalling.

## Data Version Control

As second step, you need to download your raw data and datasets. This project
uses the Data Version Control [DVC](https://dvc.org/) library to organize and
version all datasets and data.

For a tutorial about how to use `dvc` you can use the guide at DVC:
[Get Started: Data Management](https://dvc.org/doc/start/data-management).

```bash
pip instal -r requirements-dev.txt
dvc remote modify --local deep-cloud user {your-dvc-remote-user}
dvc remote modify --local deep-cloud password {your-dvc-remote-password}
dvc pull  # Download data from your dvc remote storage
```

> If your provider does not provide webdav, check the alternatives offered at
> [DVC Remotes](https://dvc.org/doc/user-guide/data-management/remote-storage)
> and your storage provider.

Make sure your MNIST data repository provides a list of `npz` datasets at the
folder configured as _DEMO_ADVANCED_DATA_PATH_ in your environment
(default: `./data`). The API metadata will provide all the file name of the
datasets available in that folder.

> Note model function `demo_advanced.train` expects `npz` files with keys
> `images` and `labels`.

## FLFlow Experiments and Models Registry

This example uses [MLFlow](https://mlflow.org/) to track and store models.
Users can use the API argument `mlflow_uri` to indicate the mlflow instance
where they want to store their training metrics, models and artifacts.

> TODO: Implement basic authorization on API methods to login into the MLFlow
> server forwarding the username and password provided by the user.

By default, the argument `mlflow_uri` is required, however, you can set up
the following environment variables to define default values in case you want
to provide your users with a public MLFlow server or experiments:

- _MLFLOW_TRACKING_URI_ pointing to the MLflow server to use for storing models.
- _MLFLOW_TRACKING_USERNAME_ username to use with HTTP Basic authentication on MLflow.
- _MLFLOW_TRACKING_PASSWORD_ password to use with HTTP Basic authentication on MLflow.
- _MLFLOW_EXPERIMENT_NAME_ experiment identification to store trainings on MLflow.
- _MLFLOW_EXPERIMENT_ID_ alternative to _MLFLOW_EXPERIMENT_NAME_ by on MLflow.

We recommend to create an `.env` file from `.env.sample.` to load by your
script. Additionally you can directly use the command line:

```bash
export MLFLOW_TRACKING_USERNAME={your-mlflow-username}
export MLFLOW_TRACKING_PASSWORD={your-mlflow-password}
export MLFLOW_TRACKING_URI=https://{your-mlflow-address}:{mlflow-port}
export MLFLOW_EXPERIMENT_ID={your-mlflow-experiment-id}
```

> Username and password are only required on MLFlow deployments protected by
> user and password.

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

- _DEMO_ADVANCED_DATA_PATH_ pointing to the training datasets, default `./data`.

Model data configuration environment variables:

- _DEMO_ADVANCED_LABEL_DIMENSIONS_ dimensions the labels are hot encoded, default `10`.
- _DEMO_ADVANCED_IMAGE_SIZE_ vertical and horizontal pixels per image, default `28`.

## Flows and deployments

This example uses [Prefect](https://docs.prefect.io/) to define and run flow
processes to automate training, dataset updates and possible deployments of
the model.

> TODO: Add flow and deployment to update datasets from public repository.  
> TODO: Add flow and deployment to deploy model on a public server.

Flows are defined inside the `flows` folder. You can run them using the
prefect CLI or from a prefect server deployment. For example, to run the
training flow you can use the following commands:

```bash
$ prefect deploy --params={training-arguments} flows/training.py:main
```

Note that in case you are running prefect CLI or server, you need to install
and configure a database backend. For more information, check the Prefect
docs [external-requirements](https://docs.prefect.io/2.10.18/getting-started/installation/#external-requirements).
In case you are using `conda` you can install SQLite with the following
command:

```bash
$ conda install -c conda-forge sqlite
```

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

Tests are performed by a remote model named `demo-advanced-testing` using its
version `1`. In order to pass all tests, you need to provide this model on
your MLFlow model registry and configure the environment variables to access
it. Experiments can be tracked if you set a _MLFLOW_EXPERIMENT_NAME_ or
_MLFLOW_EXPERIMENT_ID_, leaving empty those variables prevents the generation
of experiment tracking on the MLFlow server.

> If tests fail with `Registered Model with name=demo-advanced-testing not found`
> but you are sure that the model `demo-advanced-testing` exists in your MLFlow
> registry, ensure your MLFlow environment is accessible at testing runtime.

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
├── demo_advanced           <- Package folder containing the model code
├── deployments             <- Folder with prefect deployment configurations
├── dvc.lock                <- Data record and output state tracking (dvc)
├── dvc.yaml                <- Configuration and stages for dvc
├── flows                   <- Folder with Prefect flows
├── pyproject.toml          <- Makes project installable (pip install -e .)
├── htmlcov                 <- Report from tox qc.cov environment
├── requirements-test.txt   <- Requirements file for testing the service
├── requirements.txt        <- Requirements file for running the service
├── tests                   <- Folder containing tests for the API methods
└── tox.ini                 <- Automated testing configuration file for tox
```
