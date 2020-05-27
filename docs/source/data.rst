:github_url: https://github.com/Project-MONAI/MONAI

.. _data:

Data
====

Generic Interfaces
------------------
.. currentmodule:: monai.data

`Dataset`
~~~~~~~~~
.. autoclass:: Dataset
  :members:
  :special-members: __getitem__

`PersistentDataset`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: PersistentDataset
  :members:
  :special-members: __getitem__

`CacheDataset`
~~~~~~~~~~~~~~
.. autoclass:: CacheDataset
  :members:
  :special-members: __getitem__

`ZipDataset`
~~~~~~~~~~~~
.. autoclass:: ZipDataset
  :members:
  :special-members: __getitem__

`ArrayDataset`
~~~~~~~~~~~~~~
.. autoclass:: ArrayDataset
  :members:
  :special-members: __getitem__


Patch-based dataset
-------------------

`GridPatchDataset`
~~~~~~~~~~~~~~~~~~
.. autoclass:: GridPatchDataset
  :members:


Nifti format handling
---------------------

Reading
~~~~~~~
.. automodule:: monai.data.NiftiDataset
  :members:

Writing Nifti
~~~~~~~~~~~~~
.. automodule:: monai.data.NiftiSaver
  :members:

.. automodule:: monai.data.write_nifti
  :members:


PNG format handling
-------------------

Writing PNG
~~~~~~~~~~~
.. automodule:: monai.data.PNGSaver
  :members:

.. automodule:: monai.data.write_png
  :members:


Synthetic
---------
.. automodule:: monai.data.synthetic
  :members:


Utilities
---------
.. automodule:: monai.data.utils
  :members:

