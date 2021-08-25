:github_url: https://github.com/Project-MONAI/MONAI

.. _metrics:

Metrics
=======
.. currentmodule:: monai.metrics

`FROC`
------
.. autofunction:: compute_froc_score

`Metric`
--------
.. autoclass:: Metric
    :members:

`IterationMetric`
-----------------
.. autoclass:: IterationMetric
    :members:

`Cumulative`
------------
.. autoclass:: Cumulative
    :members:

`CumulativeIterationMetric`
---------------------------
.. autoclass:: CumulativeIterationMetric
    :members:

`Mean Dice`
-----------
.. autofunction:: compute_meandice

.. autoclass:: DiceMetric
    :members:

`Area under the ROC curve`
--------------------------
.. autofunction:: compute_roc_auc

.. autoclass:: ROCAUCMetric
    :members:

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

`Mean squared error`
--------------------
.. autoclass:: MSEMetric
    :members:

`Mean absolute error`
---------------------
.. autoclass:: MAEMetric
    :members:

`Root mean squared error`
-------------------------
.. autoclass:: RMSEMetric
    :members:

`Peak signal to noise ratio`
----------------------------
.. autoclass:: PSNRMetric
    :members:
