:github_url: https://github.com/Project-MONAI/MONAI

.. MONAI documentation master file, created by
   sphinx-quickstart on Wed Feb  5 09:40:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Project MONAI
=============


*Medical Open Network for AI*

MONAI is a `PyTorch <https://pytorch.org/>`_-based, `open-source <https://github.com/Project-MONAI/MONAI/blob/master/LICENSE>`_ framework
for deep learning in healthcare imaging, part of `PyTorch Ecosystem <https://pytorch.org/ecosystem/>`_.

Its ambitions are:

- developing a community of academic, industrial and clinical researchers collaborating on a common foundation;
- creating state-of-the-art, end-to-end training workflows for healthcare imaging;
- providing researchers with the optimized and standardized way to create and evaluate deep learning models.

Features
--------

*The codebase is currently under active development*

- flexible pre-processing for multi-dimensional medical imaging data;
- compositional & portable APIs for ease of integration in existing workflows;
- domain-specific implementations for networks, losses, evaluation metrics and more;
- customizable design for varying user expertise;
- multi-GPU data parallelism support.


Getting started
---------------

Tutorials & examples are located at `monai/examples <https://github.com/Project-MONAI/MONAI/tree/master/examples>`_.

Technical documentation is available via `Read the Docs <https://monai.readthedocs.io/en/latest/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Technical highlights

   highlights.md

.. toctree::
   :maxdepth: 1
   :caption: APIs

   transforms
   losses
   networks
   metrics
   data
   engines
   inferers
   handlers
   visualize
   utils

.. toctree::
  :maxdepth: 1
  :caption: Installation

  installation


Contributing
------------

For guidance on making a contribution to MONAI, see the `contributing guidelines
<https://github.com/Project-MONAI/MONAI/blob/master/CONTRIBUTING.md>`_.


Links
-----

- Website: https://monai.io/
- API documentation: https://monai.readthedocs.io/en/latest/
- Code: https://github.com/Project-MONAI/MONAI
- Project tracker: https://github.com/Project-MONAI/MONAI/projects
- Issue tracker: https://github.com/Project-MONAI/MONAI/issues
- Changelog: https://github.com/Project-MONAI/MONAI/blob/master/CHANGELOG.md
- Wiki: https://github.com/Project-MONAI/MONAI/wiki
- FAQ: https://github.com/Project-MONAI/MONAI/wiki/Frequently-asked-questions-and-answers
- Test status: https://github.com/Project-MONAI/MONAI/actions
- PyPI package: https://pypi.org/project/monai/
- Docker Hub: https://hub.docker.com/r/projectmonai/monai
- Google Group: https://groups.google.com/forum/#!forum/project-monai
- Reddit: https://www.reddit.com/r/projectmonai/


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
