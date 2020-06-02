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
.. autoclass:: monai.data.NiftiDataset
  :members:

Writing Nifti
~~~~~~~~~~~~~
.. autoclass:: monai.data.NiftiSaver
  :members:

.. autofunction:: monai.data.write_nifti


PNG format handling
-------------------

Writing PNG
~~~~~~~~~~~
.. autoclass:: monai.data.PNGSaver
  :members:

.. autofunction:: monai.data.write_png


Synthetic
---------
.. automodule:: monai.data.synthetic
  :members:


Utilities
---------
.. automodule:: monai.data.utils
  :members:


Decathalon Datalist
~~~~~~~~~~~~~~~~~~~
.. autofunction:: monai.data.load_decathalon_datalist


DataLoader
~~~~~~~~~~
.. autofunction:: monai.data.DataLoader
