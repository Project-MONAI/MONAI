:github_url: https://github.com/Project-MONAI/MONAI

Lazy Resampling
===============

.. toctree::
   : maxdepth: 2

    mb_specification
    config_syntax.md

Introduction
^^^^^^^^^^^^

Lazy Resampling is a new feature for MONAI 1.2. This feature is still experimental at this time and it is possible that
behaviour and APIs will change in upcoming releases.

Lazy resampling is a feature that can be used to improve preprocessing pipelines in the following ways:
 * it can improve pipeline execution time
 * it can improve pipeline memory usage
 * it can improve image and segmentation quality by reducing incidental noise caused by resampling

How Lazy Resampling changes preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to understand how lazy resampling changes preprocessing, we'll first discuss standard processing pipeline
behaviour, and then compare it with the way lazy resampling works.

Traditional resampling pipelines
++++++++++++++++++++++++++++++++

With traditional resampling, found both in MONAI and many other preprocessing libraries, you typically define a sequence
of transforms and pass them to a ``Compose`` object, such as `monai.transforms.compose.Compose`_.

Example::

    transforms = [
        LoadImaged(keys=["img", "seg"], ...),
        EnsureChannelFirstd(keys=["img", "seg"], ...),
        Spacingd(keys=["img", "seg"], ...),
        Orientationd(keys=["img", "seg"], ...),
        RandSpatialCropd(keys=["img", "seg"], ...),
        RandRotate90d(keys=["img", "seg"], ...),
        RandRotated(keys=["img", "seg"], ...),
        RandZoomd(keys=["img", "seg"], ...),
        RandGaussianNoised(keys="img", ...),
    ]
    compose = Compose(transforms)

    # elsewhere this will be called many times (such as in a Dataset instance)
    outputs = compose(inputs)
::

The following will then happen when we call ``compose(inputs)``:

1. ``LoadImaged`` is called with its inputs (a dictionary of strings containing file locations). This loads and
   returns a dictionary of the corresponding data samples
2. ``EnsureChannelFirstd`` is called with the dictionary of data samples and adds a channel so that they have the
   appropriate shape for the rest of the pipeline
3. ``Spacingd`` is called and reinterpolates the data samples
4. ``Orientationd`` permutes the data samples so that their spatial dimensions are reorganised
5. ``RandSpatialCropd`` crops a random patch of the data samples, throwing away the rest of the data in the process
6. ``RandRotate90d`` has a chance of performing a tensor-based rotation of the data samples
7. ``RandRotated`` has a chance of performing a full resample of the data samples
8. ``RandZoomd`` has a chance of performing a reinterpolation of the data samples
9. ``RandGaussianNoised`` has a chance of adding noise to ``img``

Overall, there are up to three occasions where the data is either interpolated or resampled through spatial transforms.
Furthermore, the crop that occurs means that the output data samples might contain pixels for which there is data but
that show padding values, because the data was thrown away by ``RandSpatialCrop``.

Each of these operations takes time and memory, but, as we can see in the example above, also creates resampling
artifacts and can even destroy data in the resulting data samples (see `lazy resampling best practices`_ for examples).

Lazy resampling pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^

Lazy resampling works very differently. When you execute the same pipeline with `lazy=True`, the following happens:

1. ``LoadImaged`` behaves identically
2. ``EnsureChannelFirstd`` behaves identically
3. ``Spacingd`` is executing lazily. It puts a description of the operation that it wants to perform onto a list of
   pending operations
4. ``Orientationd`` is executing lazily. It adds a description of its own operation to the pending operation list so
   now there are 2 pending operations
5. ``RandSpatialCropd`` is executing lazily. It adds a description of its own operation to the pending operation list
   so now there are 3 pending operations
6. ``RandRotate90d`` is executing lazily. It adds a description of its own operation to the pending operation list
   so now there are 4 pending operations
7. ``RandRotated`` is executing lazily. It adds a description of its own operation to the pending operation list
   so now there are 5 pending operations
8. ``RandZoomd`` is executing lazily. It adds a description of its own operation to the pending operation list
   so now there are 6 pending operations
   1. ``[Spacingd, Orientationd, RandSpatialCropd, RandRotate90d, RandRotated, RandZoomd]`` are all on the pending
      operations list but have yet to be carried out on the data
9. ``RandGaussianNoised`` is not a lazy transform. It is now time for the pending operations to be evaluated. Their
   descriptions are mathematically composited together, to determine the operation that results from all of them
   being carried out. This is then applied in a single resample operation. Once that is done, ``RandGaussianNoised``
   operates on the resulting data

The single resampling operation has less noise induced by resampling, as it only occurs once in this pipeline rather
than three times in the traditional pipeline. More importantly, although
