# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A collection of generic traits for MONAI transforms.
"""

from __future__ import annotations

__all__ = ["LazyTrait", "InvertibleTrait", "RandomizableTrait", "MultiSampleTrait", "ThreadUnsafe"]

from typing import Any


class LazyTrait:
    """
    An interface to indicate that the transform has the capability to execute using
    MONAI's lazy resampling feature. In order to do this, the implementing class needs
    to be able to describe its operation as an affine matrix or grid with accompanying metadata.
    This interface can be extended from by people adapting transforms to the MONAI framework as
    well as by implementors of MONAI transforms.
    """

    @property
    def lazy(self):
        """
        Get whether lazy evaluation is enabled for this transform instance.
        Returns:
            True if the transform is operating in a lazy fashion, False if not.
        """
        raise NotImplementedError()

    @lazy.setter
    def lazy(self, enabled: bool):
        """
        Set whether lazy evaluation is enabled for this transform instance.
        Args:
            enabled: True if the transform should operate in a lazy fashion, False if not.
        """
        raise NotImplementedError()

    @property
    def requires_current_data(self):
        """
        Get whether the transform requires the input data to be up to date before the transform executes.
        Such transforms can still execute lazily by adding pending operations to the output tensors.
        Returns:
            True if the transform requires its inputs to be up to date and False if it does not
        """


class InvertibleTrait:
    """
    An interface to indicate that the transform can be inverted, i.e. undone by performing
    the inverse of the operation performed during `__call__`.
    """

    def inverse(self, data: Any) -> Any:
        raise NotImplementedError()


class RandomizableTrait:
    """
    An interface to indicate that the transform has the capability to perform
    randomized transforms to the data that it is called upon. This interface
    can be extended from by people adapting transforms to the MONAI framework as well as by
    implementors of MONAI transforms.
    """

    pass


class MultiSampleTrait:
    """
    An interface to indicate that the transform has the capability to return multiple samples
    given an input, such as when performing random crops of a sample. This interface can be
    extended from by people adapting transforms to the MONAI framework as well as by implementors
    of MONAI transforms.
    """

    pass


class ThreadUnsafe:
    """
    A class to denote that the transform will mutate its member variables,
    when being applied. Transforms inheriting this class should be used
    cautiously in a multi-thread context.

    This type is typically used by :py:class:`monai.data.CacheDataset` and
    its extensions, where the transform cache is built with multiple threads.
    """

    pass
