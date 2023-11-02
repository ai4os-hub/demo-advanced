# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone, i.e. test (default: master)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# [!] Note: For the Jenkins CI/CD pipeline, input args are defined inside the
# Jenkinsfile, not here!

ARG tag=2.13.0-gpu

# Base image, e.g. tensorflow/tensorflow:2.x.x-gpu
FROM tensorflow/tensorflow:${tag}

LABEL maintainer='Borja Esteban'
LABEL version='0.0.0'
# DEEPaaS full demo/template.

# What user branch to clone [!]
ARG branch=main

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs)
#   (it is already installed in tensorflow images)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Update python packages
# [!] Remember: DEEP API V2 only works with python>=3.6
RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# EXPERIMENTAL: install deep-start script
# N.B.: This repository also contains run_jupyter.sh
RUN git clone https://github.com/deephdc/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start && \
    ln -s /srv/.deep-start/run_jupyter.sh /usr/local/bin/run_jupyter

# Install JupyterLab
ENV JUPYTER_CONFIG_DIR /srv/.deep-start/
# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash
RUN pip3 install --no-cache-dir jupyterlab

# Install Data Version Control
RUN pip3 install --no-cache-dir dvc dvc-webdav

# Install user app
RUN git clone --depth 1 -b $branch https://git.scc.kit.edu/m-team/ai/demo-advanced-api && \
    pip3 install --no-cache-dir -e ./demo-advanced-api

# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Launch deepaas
ENTRYPOINT [ "deep-start" ]
CMD ["--deepaas"]
