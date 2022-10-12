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

.. autoclass:: TciaDataset
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

.. automodule:: monai.apps.pathology.transforms.post.array
.. autoclass:: GenerateSuccinctContour
    :members:
.. autoclass:: PostProcessHoVerNetOutput
    :members:

.. automodule:: monai.apps.pathology.transforms.post.dictionary
.. autoclass:: GenerateSuccinctContourd
    :members:
.. autoclass:: PostProcessHoVerNetOutputd
    :members:

`Detection`
-----------

`Hard Negative Sampler`
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.hard_negative_sampler
    :members:

`RetinaNet Network`
~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.networks.retinanet_network
    :members:

`RetinaNet Detector`
~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.networks.retinanet_detector
    :members:

`Transforms`
~~~~~~~~~~~~
.. automodule:: monai.apps.detection.transforms.box_ops
    :members:
.. automodule:: monai.apps.detection.transforms.array
    :members:
.. automodule:: monai.apps.detection.transforms.dictionary
    :members:

`Anchor`
~~~~~~~~
.. automodule:: monai.apps.detection.utils.anchor_utils
    :members:

`Matcher`
~~~~~~~~~
.. automodule:: monai.apps.detection.utils.ATSS_matcher
    :members:

`Box coder`
~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.box_coder
    :members:

`Detection Utilities`
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.detector_utils
    :members:

.. automodule:: monai.apps.detection.utils.predict_utils
    :members:

`Inference box selector`
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.utils.box_selector
    :members:

`Detection metrics`
~~~~~~~~~~~~~~~~~~~
.. automodule:: monai.apps.detection.metrics.coco
    :members:
.. automodule:: monai.apps.detection.metrics.matching
    :members:

`Reconstruction`
----------------

FastMRIReader
~~~~~~~~~~~~~
.. autoclass:: monai.apps.reconstruction.fastmri_reader.FastMRIReader
  :members:

`ConvertToTensorComplex`
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: monai.apps.reconstruction.complex_utils.convert_to_tensor_complex

`ComplexAbs`
~~~~~~~~~~~~
.. autofunction:: monai.apps.reconstruction.complex_utils.complex_abs

`RootSumOfSquares`
~~~~~~~~~~~~~~~~~~
.. autofunction:: monai.apps.reconstruction.mri_utils.root_sum_of_squares

`ComplexMul`
~~~~~~~~~~~~
.. autofunction:: monai.apps.reconstruction.complex_utils.complex_mul

`ComplexConj`
~~~~~~~~~~~~~
.. autofunction:: monai.apps.reconstruction.complex_utils.complex_conj

`auto3dseg`
-----------

.. automodule:: monai.apps.auto3dseg
  :members:
  :imported-members:
