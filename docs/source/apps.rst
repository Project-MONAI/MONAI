:github_url: https://github.com/Project-MONAI/MONAI

.. _apps:

Applications
============
.. currentmodule:: monai.apps

`Datasets`
----------

.. autoclass:: MedNISTDataset
    :members:

.. autoclass:: DecathlonDataset
    :members:

.. autoclass:: CrossValidation
    :members:

`Utilities`
-----------

.. autofunction:: check_hash

.. autofunction:: download_url

.. autofunction:: extractall

.. autofunction:: download_and_extract

`Deepgrow`
----------

.. automodule:: monai.apps.deepgrow.transforms
.. autoclass:: AddInitialSeedPointd
    :members:
.. autoclass:: AddGuidanceSignald
    :members:
.. autoclass:: AddRandomGuidanced
    :members:
.. autoclass:: AddGuidanceFromPointsd
    :members:
.. autoclass:: SpatialCropForegroundd
    :members:
.. autoclass:: SpatialCropGuidanced
    :members:
.. autoclass:: RestoreCroppedLabeld
    :members:
.. autoclass:: FindDiscrepancyRegionsd
    :members:
.. autoclass:: FindAllValidSlicesd
    :members:
.. autoclass:: Fetch2DSliced
    :members:
