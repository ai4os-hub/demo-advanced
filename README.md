# DEEPaaS Full Template

>TODO: Complete build status
[![Build Status]()]()

To launch it, first install the package and prepare your environment then run [deepaas](https://github.com/indigo-dc/DEEPaaS) from the application client:
> Use editable mode `pip install -e .` when you are testing your patch to a package through another project.

```bash
git clone https://github.com/.../{repostory}  # Download repository
cd {repository}  # Navigate inside repository project
pip install -e .  # Install repository project
deepaas-run --listen-ip 0.0.0.0  # You deploy using deepaas
```

For more information about
Additionally you can configure the following environment variables for customization:

 - *ENV_VARIABLE_1* [**Required**], description for a required environment variable.
 - *ENV_VARIABLE_2* [**Optional**], description for an optional environment variable.
    Default: `some_value`.
 - TODO: Complete with more environment variables from your project.

The associated Docker container for this module can be found in {}. >TODO: add url

## Project structure

```
├── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                <- Project API license file
├── README.md              <- The top-level README for developers using this project.
├── VERSION                <- File indicating the API/Package version
├── api                    <- Python DEEPaaS API template modules
│   ├── __init__.py           <- File for initializing the python library
│   ├── api_v1.py             <- API core module for endpoint methods
│   ├── ...
│   ├── config                <- API configuration subpackage
│   ├── fields.py             <- API core fields for arguments
│   ├── parsers.py            <- API core for parsers and content types
│   └── utils.py              <- API utilities module
├── models                 <- Folder to store local cached models
│   └── {submodel}            <- Folder to store submodels folders
│       └── {head_task}_{timestamp}    <- Folder to store submodel data
├── pyproject.toml         <- Makes project pip installable (pip install -e .)
├── requirements-test.txt  <- Requirements file for testing the service
├── requirements.txt       <- Requirements file for running the service
├── src                    <- Package folder containing the model code
│   └── {model_code}          <- Model folder containing the model code
├── tests                  <- Folder containing tests for the API methods
│   ├── conftest.py           <- Module for test fixtures and parametrization
│   ├── test_metadata.py      <- Module for testing API get_metadata
│   └── test_predictions.py   <- Module for testing API predict
│   └── test_training.py      <- Module for testing API train
└── tox.ini                <- Generic virtual environment configuration file
```
