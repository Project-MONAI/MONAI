:github_url: https://github.com/Project-MONAI/MONAI

.. _metrics:

Metrics
=======
.. currentmodule:: monai.metrics

`Mean Dice`
-----------
.. autofunction:: compute_meandice

.. autoclass:: DiceMetric
    :members:

`Area under the ROC curve`
--------------------------
.. autofunction:: compute_roc_auc

`Confusion Matrix`
------------------
.. autofunction:: get_confusion_matrix

.. autoclass:: ConfusionMatrixMetric
    :members:

`Hausdorff Distance`
--------------------
.. autofunction:: compute_hausdorff_distance

.. autoclass:: HausdorffDistance
    :members:

`Average Surface Distance`
--------------------------
.. autofunction:: compute_average_surface_distance

.. autoclass:: SurfaceDistance
    :members:

`Occlusion sensitivity`
-----------------------
.. autofunction:: compute_occlusion_sensitivity