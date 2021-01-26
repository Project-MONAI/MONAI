:github_url: https://github.com/Project-MONAI/MONAI

.. _metrics:

Metrics
=======
.. currentmodule:: monai.metrics

`FROC`
------
.. autofunction:: compute_FROC_Score

`Mean Dice`
-----------
.. autofunction:: compute_meandice

.. autoclass:: DiceMetric
    :members:

`Area under the ROC curve`
--------------------------
.. autofunction:: compute_roc_auc

`Confusion matrix`
------------------
.. autofunction:: get_confusion_matrix

.. autoclass:: ConfusionMatrixMetric
    :members:

`Hausdorff distance`
--------------------
.. autofunction:: compute_hausdorff_distance

.. autoclass:: HausdorffDistanceMetric
    :members:

`Average surface distance`
--------------------------
.. autofunction:: compute_average_surface_distance

.. autoclass:: SurfaceDistanceMetric
    :members:
