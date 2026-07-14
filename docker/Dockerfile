FROM ubuntu:22.04

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# Install pip and git with apt
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN python3 -m pip install --no-cache-dir pip setuptools --upgrade

ARG REPO_BRANCH="main"

WORKDIR /repo

# Clone the simulation repo and install it. `pip install .` resolves every
# dependency from pyproject.toml, including the pinned EnsembleControl release
# archived on Zenodo (doi:10.5281/zenodo.21328270).
RUN git clone --branch ${REPO_BRANCH} --single-branch \
        https://github.com/milzj/saa-control-clt-simulations.git && \
    cd saa-control-clt-simulations && \
    python3 -m pip install --no-cache-dir . && \
    rm -rf /tmp

WORKDIR /repo/saa-control-clt-simulations
