# Installation Guide

## Table of Contents
1. [From PyPI](#from-pypi)
	1. [Milestone release](#milestone-release)
	2. [Weekly preview release](#weekly-preview-release)
1. [From conda-forge](#from-conda-forge)
2. [From GitHub](#from-github)
	1. [System-wide](#milestone-release)
	2. [Editable](#weekly-preview-release)
3. [Validating the install](#validating-the-install)
4. [MONAI version string](#monai-version-string)
5. [From DockerHub](#from-dockerhub)
6. [Installing the recommended dependencies](#installing-the-recommended-dependencies)

---

MONAI's core functionality is written in Python 3 (>= 3.7) and only requires [Numpy](https://numpy.org/) and [Pytorch](https://pytorch.org/).

The package is currently distributed via Github as the primary source code repository,
and the Python package index (PyPI). The pre-built Docker images are made available on DockerHub.

To install optional features such as handling the NIfTI files using
[Nibabel](https://nipy.org/nibabel/), or building workflows using [Pytorch
Ignite](https://pytorch.org/ignite/), please follow the instructions:
- [Installing the recommended dependencies](#installing-the-recommended-dependencies)

---


## From PyPI

### Milestone release
To install the [current milestone release](https://pypi.org/project/monai/):
```bash
pip install monai
```

### Weekly preview release
To install the [weekly preview release](https://pypi.org/project/monai-weekly/):
```bash
pip install monai-weekly
```

The weekly build is released to PyPI every Sunday with a pre-release build number `dev[%y%U]`.
To report any issues on the weekly preview, please include the version and commit information:
```bash
python -c "import monai; print(monai.__version__); print(monai.__commit_id__)"
```

## From conda-forge

To install the [current milestone release](https://anaconda.org/conda-forge/monai):
```bash
conda install -c conda-forge monai
```

## From GitHub
(_If you have installed the
PyPI release version using ``pip install monai``, please run ``pip uninstall
monai`` before using the commands from this section. Because ``pip`` by
default prefers the milestone release_.)

The milestone versions are currently planned and released every few months.  As the
codebase is under active development, you may want to install MONAI from GitHub
for the latest features:

### Option 1 (as a part of your system-wide module):
```bash
pip install git+https://github.com/Project-MONAI/MONAI#egg=monai
```
or, to build with MONAI Cpp/CUDA extensions:
```bash
BUILD_MONAI=1 pip install git+https://github.com/Project-MONAI/MONAI#egg=monai
```

To build the extensions, if the system environment already has a version of Pytorch installed,
`--no-build-isolation` might be preferred:
```bash
BUILD_MONAI=1 pip install --no-build-isolation git+https://github.com/Project-MONAI/MONAI#egg=monai
```
this command will download and install the current `dev` branch of [MONAI from
GitHub](https://github.com/Project-MONAI/MONAI).

This documentation website by default shows the information for the latest version.

### Option 2 (editable installation):
To install an editable version of MONAI, it is recommended to clone the codebase directly:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
```
This command will create a ``MONAI/`` folder in your current directory.
You can install it by running:
```bash
cd MONAI/
python setup.py develop
```
or, to build with MONAI Cpp/CUDA extensions and install:
```bash
cd MONAI/
BUILD_MONAI=1 python setup.py develop
# for MacOS
BUILD_MONAI=1 CC=clang CXX=clang++ python setup.py develop
```

To uninstall the package please run:
```bash
cd MONAI/
python setup.py develop --uninstall

# to further clean up the MONAI/ folder (Bash script)
./runtests.sh --clean
```

Alternatively, simply adding the root directory of the cloned source code (e.g., ``/workspace/Documents/MONAI``) to your ``$PYTHONPATH``
and the codebase is ready to use (without the additional features of MONAI C++/CUDA extensions).


## Validating the install
You can verify the installation by:
```bash
python -c 'import monai; monai.config.print_config()'
```
If the installation is successful, this command will print out the MONAI version information, and this confirms the core
modules of MONAI are ready-to-use.


## MONAI version string
The MONAI version string shows the current status of your local installation. For example:
```
MONAI version: 0.1.0+144.g52c763d.dirty
```
- ``0.1.0`` indicates that your installation is based on the ``0.1.0`` milestone release.
- ``+144`` indicates that your installation is 144 git commits ahead of the milestone release.
- ``g52c763d`` indicates that your installation corresponds to the git commit hash ``52c763d``.
- ``dirty`` indicates that you have modified the codebase locally, and the codebase is inconsistent with ``52c763d``.


## From DockerHub
Make sure you have installed the NVIDIA driver and Docker 19.03+ for your Linux distribution.
Note that you do not need to install the CUDA toolkit on the host, but the driver needs to be installed.
Please find out more information on [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Assuming that you have the Nvidia driver and Docker 19.03+ installed, running the following command will
download and start a container with the latest version of MONAI. The latest `dev` branch of MONAI from GitHub
is included in the image.
```bash
docker run --gpus all --rm -ti --ipc=host projectmonai/monai:latest
```

You can also run a milestone release docker image by specifying the image tag, for example:
```
docker run --gpus all --rm -ti --ipc=host projectmonai/monai:0.1.0
```

## Installing the recommended dependencies
By default, the installation steps will only download and install the minimal requirements of MONAI.
Optional dependencies can be installed using [the extras syntax](https://packaging.python.org/tutorials/installing-packages/#installing-setuptools-extras) to support additional features.

For example, to install MONAI with Nibabel and Scikit-image support:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -e '.[nibabel,skimage]'
```

Alternatively, to install all optional dependencies:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -e '.[all]'
```

To install all optional dependencies with `pip` based on MONAI development environment settings:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
pip install -r requirements-dev.txt
```

To install all optional dependencies with `conda` based on MONAI development environment settings:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd MONAI/
conda create -n <name> python=<ver>  # eg 3.9
conda env update -n <name> -f environment-dev.yml
```

Since MONAI v0.2.0, the extras syntax such as `pip install 'monai[nibabel]'` is available via PyPI.

- The options are
```
[nibabel, skimage, pillow, tensorboard, gdown, ignite, torchvision, itk, tqdm, lmdb, psutil, cucim, openslide, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs, pyyaml, fire, jsonschema, pynrrd]
```
which correspond to `nibabel`, `scikit-image`, `pillow`, `tensorboard`,
`gdown`, `pytorch-ignite`, `torchvision`, `itk`, `tqdm`, `lmdb`, `psutil`, `cucim`, `openslide-python`, `pandas`, `einops`, `transformers`, `mlflow`, `matplotlib`, `tensorboardX`, `tifffile`, `imagecodecs`, `pyyaml`, `fire`, `jsonschema`, `pynrrd`, `scipy` ,respectively.

- `pip install 'monai[all]'` installs all the optional dependencies.
