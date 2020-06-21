<p align="center">
  <img src="https://github.com/Project-MONAI/MONAI/raw/master/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

**M**edical **O**pen **N**etwork for **AI**

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAI/workflows/build/badge.svg?branch=master)](https://github.com/Project-MONAI/MONAI/commits/master)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://monai.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAI/branch/master/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAI)
[![PyPI version](https://badge.fury.io/py/monai.svg)](https://badge.fury.io/py/monai)

MONAI is a [PyTorch](https://pytorch.org/)-based, [open-source](https://github.com/Project-MONAI/MONAI/blob/master/LICENSE) framework for deep learning in healthcare imaging, part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/).
Its ambitions are:
- developing a community of academic, industrial and clinical researchers collaborating on a common foundation;
- creating state-of-the-art, end-to-end training workflows for healthcare imaging;
- providing researchers with the optimized and standardized way to create and evaluate deep learning models.


## Features
> _The codebase is currently under active development._

- flexible pre-processing for multi-dimensional medical imaging data;
- compositional & portable APIs for ease of integration in existing workflows;
- domain-specific implementations for networks, losses, evaluation metrics and more;
- customizable design for varying user expertise;
- multi-GPU data parallelism support.

## Installation
To install [the current release](https://pypi.org/project/monai/):
```bash
pip install monai
```

To install from the source code repository:
```bash
pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
```

Alternatively, pre-built Docker image is available via [DockerHub](https://hub.docker.com/r/projectmonai/monai):
  ```bash
  # with docker v19.03+
  docker run --gpus all --rm -ti --ipc=host projectmonai/monai:latest
  ```

For more details, please refer to [the installation guide](https://monai.readthedocs.io/en/latest/installation.html).

## Getting Started

Tutorials & examples are located at [monai/examples](https://github.com/Project-MONAI/MONAI/tree/master/examples).

Technical documentation is available via [Read the Docs](https://monai.readthedocs.io/en/latest/).

## Contributing
For guidance on making a contribution to MONAI, see the [contributing guidelines](https://github.com/Project-MONAI/MONAI/blob/master/CONTRIBUTING.md).

## Links
- Website: https://monai.io/
- API documentation: https://monai.readthedocs.io/en/latest/
- Code: https://github.com/Project-MONAI/MONAI
- Project tracker: https://github.com/Project-MONAI/MONAI/projects
- Issue tracker: https://github.com/Project-MONAI/MONAI/issues
- Wiki: https://github.com/Project-MONAI/MONAI/wiki
- Test status: https://github.com/Project-MONAI/MONAI/actions
