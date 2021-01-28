:github_url: https://github.com/Project-MONAI/MONAI

.. _handlers:

Event handlers
==============
.. currentmodule:: monai.handlers

Model checkpoint loader
-----------------------
.. autoclass:: CheckpointLoader
  :members:

Model checkpoint saver
----------------------
.. autoclass:: CheckpointSaver
  :members:

CSV saver
---------
.. autoclass:: ClassificationSaver
    :members:


Iteration Metric
----------------
.. autoclass:: IterationMetric
    :members:


Mean Dice metrics handler
-------------------------
.. autoclass:: MeanDice
    :members:


ROC AUC metrics handler
-----------------------
.. autoclass:: ROCAUC
    :members:


Confusion matrix metrics handler
--------------------------------
.. autoclass:: ConfusionMatrix
    :members:


Hausdorff distance metrics handler
----------------------------------
.. autoclass:: HausdorffDistance
    :members:


Surface distance metrics handler
--------------------------------
.. autoclass:: SurfaceDistance
    :members:


Metric logger
-------------
.. autoclass:: MetricLogger
    :members:


Segmentation saver
------------------
.. autoclass:: SegmentationSaver
    :members:


Training stats handler
----------------------
.. autoclass:: StatsHandler
    :members:


Tensorboard handlers
--------------------
.. autoclass:: TensorBoardStatsHandler
    :members:

.. autoclass:: TensorBoardImageHandler
    :members:


LR Schedule handler
-------------------
.. autoclass:: LrScheduleHandler
    :members:


Validation handler
------------------
.. autoclass:: ValidationHandler
    :members:

SmartCache handler
------------------
.. autoclass:: SmartCacheHandler
    :members:
