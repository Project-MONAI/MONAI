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


`Clara MMARs`
-------------
.. autofunction:: download_mmar

.. autofunction:: load_from_mmar

.. autodata:: monai.apps.MODEL_DESC
    :annotation:


`Utilities`
-----------

.. autofunction:: check_hash

.. autofunction:: download_url

.. autofunction:: extractall

.. autofunction:: download_and_extract

`Deepgrow`
----------

.. automodule:: monai.apps.deepgrow.dataset
.. autofunction:: create_dataset

.. automodule:: monai.apps.deepgrow.interaction
.. autoclass:: Interaction
    :members:

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
.. autoclass:: RestoreLabeld
    :members:
.. autoclass:: ResizeGuidanced
    :members:
.. autoclass:: FindDiscrepancyRegionsd
    :members:
.. autoclass:: FindAllValidSlicesd
    :members:
.. autoclass:: Fetch2DSliced
    :members:

`Pathology`
-----------

.. automodule:: monai.apps.pathology.data
.. autoclass:: PatchWSIDataset
    :members:
.. autoclass:: SmartCachePatchWSIDataset
    :members:
.. autoclass:: MaskedInferenceWSIDataset
    :members:

.. automodule:: monai.apps.pathology.handlers
.. autoclass:: ProbMapProducer
    :members:

.. automodule:: monai.apps.pathology.metrics
.. autoclass:: LesionFROC
    :members:

.. automodule:: monai.apps.pathology.utils
.. autofunction:: compute_multi_instance_mask
.. autofunction:: compute_isolated_tumor_cells
.. autoclass:: PathologyProbNMS
    :members:

.. automodule:: monai.apps.pathology.transforms.stain.array
.. autoclass:: ExtractHEStains
    :members:
.. autoclass:: NormalizeHEStains
    :members:

.. automodule:: monai.apps.pathology.transforms.stain.dictionary
.. autoclass:: ExtractHEStainsd
    :members:
.. autoclass:: NormalizeHEStainsd
    :members:

.. automodule:: monai.apps.pathology.transforms.spatial.array
.. autoclass:: SplitOnGrid
    :members:
.. autoclass:: TileOnGrid
    :members:

.. automodule:: monai.apps.pathology.transforms.spatial.dictionary
.. autoclass:: SplitOnGridd
    :members:
.. autoclass:: TileOnGridd
    :members:

`Detection`
-----------

`Hard Negative Sampler`
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.hard_negative_sampler

`Transforms`
~~~~~~~~~~~~
.. automodule:: monai.apps.detection.transforms.box_ops
    :members:
.. automodule:: monai.apps.detection.transforms.array
    :members:
.. automodule:: monai.apps.detection.transforms.dictionary
    :members:

`Box coder`
~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.box_coder
    :members:
