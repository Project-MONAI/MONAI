# Copyright 2020 - 2021 MONAI Consortium
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
A collection of generic interfaces for MONAI transforms.
"""

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["Transform"]


class Transform(ABC):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and may modify ``data`` in place,
    the implementation should be aware of:

        #. thread safety when mutating its own states.
           When used from a multi-process context, transform's instance variables are read-only.
        #. ``data`` content unused by this transform may still be used in the
           subsequent transforms in a composed transform.
        #. storing too much information in ``data`` may not scale.

    See Also

        :py:class:`monai.transforms.Compose`
    """

    @abstractmethod
    def __call__(self, data: Any):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as :py:class:`torch.utils.data.Dataset`. This method should
        return an updated version of ``data``.
        To simplify the input validations, most of the transforms assume that

        - ``data`` is a Numpy ndarray, PyTorch Tensor or string
        - the data shape can be:

          #. string data without shape, `LoadImage` transform expects file paths
          #. most of the pre-processing transforms expect: ``(num_channels, spatial_dim_1[, spatial_dim_2, ...])``,
             except that `AddChannel` expects (spatial_dim_1[, spatial_dim_2, ...]) and
             `AsChannelFirst` expects (spatial_dim_1[, spatial_dim_2, ...], num_channels)
          #. most of the post-processing transforms expect
             ``(batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])``

        - the channel dimension is not omitted even if number of channels is one

        This method can optionally take additional arguments to help execute transformation operation.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def inverse(self, data: Any):
        """
        Inverse of ``__call__``.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
