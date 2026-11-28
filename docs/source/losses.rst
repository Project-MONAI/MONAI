:github_url: https://github.com/Project-MONAI/MONAI

.. _losses:

Loss functions
==============

.. automodule:: monai.losses
.. currentmodule:: monai.losses

The losses in ``monai.losses`` are grouped below by the task they are
typically used for.

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Task
      - Losses
    * - `Segmentation`_
      - ``DiceLoss``, ``Dice``, ``dice``, ``HardL1ACELoss``, ``SoftL1ACELoss``,
        ``MaskedDiceLoss``, ``GeneralizedDiceLoss``, ``generalized_dice``,
        ``GeneralizedWassersteinDiceLoss``, ``generalized_wasserstein_dice``,
        ``DiceCELoss``, ``dice_ce``, ``DiceFocalLoss``, ``dice_focal``,
        ``GeneralizedDiceFocalLoss``, ``generalized_dice_focal``, ``FocalLoss``,
        ``TverskyLoss``, ``ContrastiveLoss``, ``BarlowTwinsLoss``, ``BoundaryLoss``,
        ``HausdorffDTLoss``, ``LogHausdorffDTLoss``, ``SoftclDiceLoss``,
        ``SoftDiceclDiceLoss``, ``NACLLoss``, ``MCCLoss``, ``AsymmetricUnifiedFocalLoss``
    * - `Classification`_
      - ``AUCMLoss``
    * - `Detection`_
      - ``BoxGIoULoss``, ``giou``
    * - `Registration`_
      - ``BendingEnergyLoss``, ``DiffusionLoss``, ``LocalNormalizedCrossCorrelationLoss``,
        ``GlobalMutualInformationLoss``
    * - `Reconstruction`_
      - ``SSIMLoss``, ``PatchAdversarialLoss``, ``PerceptualLoss``, ``JukeboxLoss``, ``SURELoss``
    * - `Loss Wrappers`_
      - ``MultiScaleLoss``, ``MaskedLoss``, ``DeepSupervisionLoss``

.. _Segmentation:

Segmentation Losses
-------------------

Losses for semantic segmentation, region-overlap and imbalanced-label objectives.

`DiceLoss`
~~~~~~~~~~
.. autoclass:: DiceLoss
    :members:

.. autoclass:: Dice
    :members:

.. autoclass:: dice
    :members:

`HardL1ACELoss`
~~~~~~~~~~~~~~~~
.. autoclass:: HardL1ACELoss
    :members:

`SoftL1ACELoss`
~~~~~~~~~~~~~~~~
.. autoclass:: SoftL1ACELoss
    :members:

`MaskedDiceLoss`
~~~~~~~~~~~~~~~~
.. autoclass:: MaskedDiceLoss
    :members:

`GeneralizedDiceLoss`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GeneralizedDiceLoss
    :members:

.. autoclass:: generalized_dice
    :members:

`GeneralizedWassersteinDiceLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GeneralizedWassersteinDiceLoss
    :members:

.. autoclass:: generalized_wasserstein_dice
    :members:

`DiceCELoss`
~~~~~~~~~~~~
.. autoclass:: DiceCELoss
    :members:

.. autoclass:: dice_ce
    :members:

`DiceFocalLoss`
~~~~~~~~~~~~~~~
.. autoclass:: DiceFocalLoss
    :members:

.. autoclass:: dice_focal
    :members:

`GeneralizedDiceFocalLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GeneralizedDiceFocalLoss
    :members:

.. autoclass:: generalized_dice_focal
    :members:

`FocalLoss`
~~~~~~~~~~~
.. autoclass:: FocalLoss
    :members:

`TverskyLoss`
~~~~~~~~~~~~~
.. autoclass:: TverskyLoss
    :members:

`ContrastiveLoss`
~~~~~~~~~~~~~~~~~
.. autoclass:: ContrastiveLoss
    :members:

`BarlowTwinsLoss`
~~~~~~~~~~~~~~~~~
.. autoclass:: BarlowTwinsLoss
    :members:

`BoundaryLoss`
~~~~~~~~~~~~~~
.. autoclass:: BoundaryLoss
    :members:

`HausdorffDTLoss`
~~~~~~~~~~~~~~~~~
.. autoclass:: HausdorffDTLoss
    :members:

.. autoclass:: LogHausdorffDTLoss
    :members:

`SoftclDiceLoss`
~~~~~~~~~~~~~~~~
.. autoclass:: SoftclDiceLoss
    :members:

`SoftDiceclDiceLoss`
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SoftDiceclDiceLoss
    :members:

`NACLLoss`
~~~~~~~~~~
.. autoclass:: NACLLoss
    :members:

`MCCLoss`
~~~~~~~~~
.. autoclass:: MCCLoss
    :members:

`AsymmetricUnifiedFocalLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: AsymmetricUnifiedFocalLoss
    :members:

.. _Classification:

Classification Losses
---------------------

Losses for image-level classification and ranking objectives.

`AUCMLoss`
~~~~~~~~~~
.. autoclass:: AUCMLoss
    :members:

.. _Detection:

Detection Losses
----------------

Losses for bounding-box regression in object detection.

`BoxGIoULoss`
~~~~~~~~~~~~~
.. autoclass:: BoxGIoULoss
    :members:

.. autoclass:: giou
    :members:

.. _Registration:

Registration Losses
-------------------

Losses for deformable image registration, measuring image similarity
and deformation-field smoothness.

`BendingEnergyLoss`
~~~~~~~~~~~~~~~~~~~
.. autoclass:: BendingEnergyLoss
    :members:

`DiffusionLoss`
~~~~~~~~~~~~~~~
.. autoclass:: DiffusionLoss
    :members:

`LocalNormalizedCrossCorrelationLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LocalNormalizedCrossCorrelationLoss
    :members:

`GlobalMutualInformationLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: GlobalMutualInformationLoss
    :members:

.. _Reconstruction:

Reconstruction Losses
---------------------

Losses for image reconstruction and restoration, including perceptual
and adversarial objectives.

`SSIMLoss`
~~~~~~~~~~
.. autoclass:: monai.losses.ssim_loss.SSIMLoss
    :members:

`PatchAdversarialLoss`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PatchAdversarialLoss
    :members:

`PerceptualLoss`
~~~~~~~~~~~~~~~~~
.. autoclass:: PerceptualLoss
    :members:

`JukeboxLoss`
~~~~~~~~~~~~~~
.. autoclass:: JukeboxLoss
    :members:

`SURELoss`
~~~~~~~~~~
.. autoclass:: SURELoss
    :members:


.. _Loss Wrappers:

Loss Wrappers
-------------

Wrappers that adapt or compose other loss functions.

`MultiScaleLoss`
~~~~~~~~~~~~~~~~
.. autoclass:: MultiScaleLoss
    :members:

`MaskedLoss`
~~~~~~~~~~~~
.. autoclass:: MaskedLoss
    :members:

`DeepSupervisionLoss`
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: DeepSupervisionLoss
    :members:
