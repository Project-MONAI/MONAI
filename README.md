<p align="center">
<img src="https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/docs/images/MONAI-logo-color.png" width="50%" alt="Project MONAI">
</p>

**MONAI: Medical Open Network for AI**

[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/monai.svg)](https://badge.fury.io/py/monai)
[![docker](https://img.shields.io/badge/docker-pull-green.svg?logo=docker&logoColor=white)](https://hub.docker.com/r/projectmonai/monai)
[![conda](https://img.shields.io/conda/vn/conda-forge/monai?color=green)](https://anaconda.org/conda-forge/monai)
[![Documentation Status](https://readthedocs.org/projects/monai/badge/?version=latest)](https://monai.readthedocs.io/en/latest/)

MONAI is a free and open-source deep learning framework for medical imaging, built on top of [PyTorch](https://pytorch.org/) and listed in the [PyTorch Ecosystem](https://pytorch.org/ecosystem/). The project aims to:

- Bring academic, industry and clinical researchers together around one shared foundation;
- Offer modern, complete training pipelines for medical imaging tasks;
- Give researchers a standard, well-optimized way to build and assess deep learning models.

## Key Capabilities

> For details, see the [technical highlights](https://monai.readthedocs.io/en/latest/highlights.html) and the [release notes](https://monai.readthedocs.io/en/latest/whatsnew.html).

- Pre-processing that can be adapted to medical images of any dimensionality;
- Modular, portable APIs that slot easily into existing code;
- Networks, loss functions, metrics and other components designed for the medical domain;
- A design that suits users with different levels of experience;
- Data-parallel training across several GPUs and several machines.

## Dependencies

MONAI runs on the Python versions that are still officially supported. It depends directly on NumPy and PyTorch, plus a number of optional packages.

* Each major release lists its own dependency versions. The `dev` branch is the in-progress version and usually tracks current dependencies, including the fixes needed to keep up with them.
* Supported PyTorch versions are the newest release and the three minor versions before it. If a PyTorch version clashes with another dependency, support for it may wait until the next major release.
* For other dependencies the project mostly follows [SPEC0](https://scientific-python.org/specs/spec-0000), supporting versions for up to about two years. A version with a known vulnerability or defect may be explicitly dropped.
* Exact version requirements are in `pyproject.toml`.

## How to Install

The latest [stable release](https://pypi.org/project/monai/) installs with a single command:

```bash
pip install monai
```

Other ways to install are covered in the [installation guide](https://monai.readthedocs.io/en/latest/installation.html).

## Where to Begin

Two interactive notebooks run directly in Colab: the [MedNIST demo](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb) and [MONAI for PyTorch Users](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/main/modules/developer_guide.ipynb).

More examples and notebooks live in the [tutorials repository](https://github.com/Project-MONAI/tutorials). Full technical documentation is at [monai.readthedocs.io](https://monai.readthedocs.io).

## Docker

A ready-made image is published on [Docker Hub](https://hub.docker.com/r/projectmonai/monai). The `latest` tag follows the current `dev` branch, and release versions have their own tags. If you want a smaller image, you can build one yourself from `Dockerfile.slim`; the instructions are inside that file.

To start a container with the newest MONAI and GPU access:

```bash
docker run -ti --rm --gpus all projectmonai/monai:latest /bin/bash
```

## Citing MONAI

If MONAI helped with your research, please give credit by citing it. A citation can be exported from <https://arxiv.org/abs/2211.02701>.

## Model Zoo

The [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) is a shared collection where the community publishes strong models. Models use the [MONAI Bundle format](https://monai.readthedocs.io/en/latest/bundle_intro.html), which makes it simple to [start building workflows](https://github.com/Project-MONAI/tutorials/tree/main/model_zoo).

## Contributing

To learn how to contribute, read the [contributing guidelines](https://github.com/Project-MONAI/MONAI/blob/dev/CONTRIBUTING.md).

## Community

Follow or talk to the project on Twitter/X ([@ProjectMONAI](https://twitter.com/ProjectMONAI)) or [LinkedIn](https://www.linkedin.com/company/projectmonai), or join the [Slack channel](https://forms.gle/QTxJq3hFictp31UM9).

Questions and answers are welcome in the [GitHub Discussions](https://github.com/Project-MONAI/MONAI/discussions) section.

## Useful Links

- Website: <https://project-monai.github.io/>
- Documentation (stable): <https://monai.readthedocs.io/>
- Documentation (latest dev): <https://monai.readthedocs.io/en/latest/>
- Source code: <https://github.com/Project-MONAI/MONAI>
- Project board: <https://github.com/Project-MONAI/MONAI/projects>
- Issues: <https://github.com/Project-MONAI/MONAI/issues>
- Wiki: <https://github.com/Project-MONAI/MONAI/wiki>
- CI status: <https://github.com/Project-MONAI/MONAI/actions>
- PyPI: <https://pypi.org/project/monai/>
- conda-forge: <https://anaconda.org/conda-forge/monai>
- Weekly preview builds: <https://pypi.org/project/monai-weekly/>
- Docker Hub: <https://hub.docker.com/r/projectmonai/monai>
