:github_url: https://github.com/Project-MONAI/MONAI

.. _data:

Data
====

Generic Interfaces
------------------
.. automodule:: monai.data.dataset
.. currentmodule:: monai.data.dataset

`Dataset`
~~~~~~~~~
.. autoclass:: Dataset
  :members:
  :special-members: __getitem__


Patch-based dataset
-------------------

`GridPatchDataset`
~~~~~~~~~~~~~~~~~~
.. automodule:: monai.data.grid_dataset
.. currentmodule:: monai.data.grid_dataset
.. autoclass:: GridPatchDataset
  :members:


Sliding window inference
------------------------

.. automodule:: monai.data.sliding_window_inference
  :members:



Nifti format handling
---------------------

Reading
~~~~~~~
.. automodule:: monai.data.nifti_reader
  :members:

Writing
~~~~~~~
.. automodule:: monai.data.nifti_saver
  :members:

.. automodule:: monai.data.nifti_writer
  :members:


Synthetic
---------
.. automodule:: monai.data.synthetic
  :members:


Utilities
---------
.. automodule:: monai.data.utils
  :members:

