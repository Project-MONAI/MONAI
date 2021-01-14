<p align="center">
  <img src="https://github.com/Project-MONAI/MONAI/raw/master/docs/images/MONAI-logo-color.png" width="50%" alt='project-monai'>
</p>

**M**edical **O**pen **N**etwork for **AI**

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Build](https://github.com/Project-MONAI/MONAI/workflows/build/badge.svg?branch=master)](https://github.com/Project-MONAI/MONAI/commits/master)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://docs.monai.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Project-MONAI/MONAI/branch/master/graph/badge.svg)](https://codecov.io/gh/Project-MONAI/MONAI)
[![PyPI version](https://badge.fury.io/py/monai.svg)](https://badge.fury.io/py/monai)

MONAI is a [PyTorch](https://pytorch.org/)-based, [open-source](https://github.com/Project-MONAI/MONAI/blob/master/LICENSE) framework for deep learning in healthcare imaging, part of [PyTorch Ecosystem](https://pytorch.org/ecosystem/).
Its ambitions are:
- developing a community of academic, industrial and clinical researchers collaborating on a common foundation;
- creating state-of-the-art, end-to-end training workflows for healthcare imaging;
- providing researchers with the optimized and standardized way to create and evaluate deep learning models.


## Features
> _The codebase is currently under active development._
> _Please see [the technical highlights](https://docs.monai.io/en/latest/highlights.html) of the current milestone release._

- flexible pre-processing for multi-dimensional medical imaging data;
- compositional & portable APIs for ease of integration in existing workflows;
- domain-specific implementations for networks, losses, evaluation metrics and more;
- customizable design for varying user expertise;
- multi-GPU data parallelism support.


## Installation

### Installing [the current release](https://pypi.org/project/monai/):
```bash
pip install monai
```

### Installing the master branch from the source code repository:
```bash
pip install git+https://github.com/Project-MONAI/MONAI#egg=MONAI
```

### Using the pre-built Docker image [DockerHub](https://hub.docker.com/r/projectmonai/monai):
  ```bash
  # with docker v19.03+
  docker run --gpus all --rm -ti --ipc=host projectmonai/monai:latest
  ```

For more details, please refer to [the installation guide](https://docs.monai.io/en/latest/installation.html).

## Getting Started

[MedNIST demo](https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe) and [MONAI for PyTorch Users](https://colab.research.google.com/drive/1boqy7ENpKrqaJoxFlbHIBnIODAs1Ih1T) are available on Colab.

Examples and notebook tutorials are located at [Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials).

Technical documentation is available at [docs.monai.io](https://docs.monai.io).

## Contributing
For guidance on making a contribution to MONAI, see the [contributing guidelines](https://github.com/Project-MONAI/MONAI/blob/master/CONTRIBUTING.md).

## Community
Join the conversation on Twitter [@ProjectMONAI](https://twitter.com/ProjectMONAI) or join our [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Ask and answer questions over on [MONAI's GitHub Discussions tab](https://github.com/Project-MONAI/MONAI/discussions).

## Links
- Website: https://monai.io/
- API documentation: https://docs.monai.io
- Code: https://github.com/Project-MONAI/MONAI
- Project tracker: https://github.com/Project-MONAI/MONAI/projects
- Issue tracker: https://github.com/Project-MONAI/MONAI/issues
- Wiki: https://github.com/Project-MONAI/MONAI/wiki
- Test status: https://github.com/Project-MONAI/MONAI/actions
