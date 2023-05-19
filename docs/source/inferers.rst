:github_url: https://github.com/Project-MONAI/MONAI

.. _inferers:

Inference methods
=================

Inferers
--------

.. currentmodule:: monai.inferers
.. autoclass:: Inferer
    :members:
    :special-members: __call__

`PatchInferer`
~~~~~~~~~~~~~~
.. autoclass:: PatchInferer
    :members:
    :special-members: __call__

`SimpleInferer`
~~~~~~~~~~~~~~~
.. autoclass:: SimpleInferer
    :members:
    :special-members: __call__

`SlidingWindowInferer`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SlidingWindowInferer
    :members:
    :special-members: __call__

`SaliencyInferer`
~~~~~~~~~~~~~~~~~
.. autoclass:: SaliencyInferer
    :members:
    :special-members: __call__

`SliceInferer`
~~~~~~~~~~~~~~
.. autoclass:: SliceInferer
    :members:
    :special-members: __call__


Splitters
---------
.. currentmodule:: monai.inferers
.. autoclass:: Splitter
    :members:
    :special-members: __call__

`SlidingWindowSplitter`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SlidingWindowSplitter
    :members:
    :special-members: __call__

Mergers
-------
.. currentmodule:: monai.inferers
.. autoclass:: Merger
    :members:
    :special-members: __call__

`AvgMerger`
~~~~~~~~~~~
.. autoclass:: AvgMerger
    :members:
    :special-members: __call__



Sliding Window Inference Function
---------------------------------

.. autofunction:: monai.inferers.sliding_window_inference
