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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from monai.utils import ensure_tuple_size

__all__ = ["Merger", "AvgMerger"]


class Merger(ABC):
    """
    A base class for merging patches.
    Extend this class to support operations for `PatchInference`, e.g. `AvgMerger`.

    Args:
        device: the device where Merger tensors should reside.
    """

    def __init__(self, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self.is_initialized = False
        self.is_finalized = False

    @abstractmethod
    def initialize(self, output_shape: Sequence[int] | None = None) -> None:
        """
        Initialize the merger.

        Args:
            output_shape: the shape of output tensor (B,C,H,W,[D])

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output
            location: a tuple/list giving the top left location of the patch in the original image.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def finalize(self) -> torch.Tensor:
        """
        Perform final operations for merging patches.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_output(self) -> torch.Tensor:
        """
        Get the output merged tensor.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class AvgMerger(Merger):
    """Merge patches by taking average of the overlapping area

    Args:
        output_shape: optional fix size for the shape of the output tensor. Default to None.
            If provided, it overwrites the `output_shape` in `initialize(output_shape)`.
        device: the device for aggregator tensors and final results.
        dtype: the dtype for aggregation and final result and .
    """

    def __init__(
        self,
        output_shape: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device, dtype)
        self.output_shape = output_shape

    def initialize(self, output_shape: Sequence[int] | None = None) -> None:
        """
        Initialize the merger by creating tensors for aggregating `values` and `counts`.

        Args:
            output_shape: the shape of output tensor (B,C,H,W,[D])
        """
        # overwrite output_shape if the self.output_shape is set
        if self.output_shape:
            output_shape = self.output_shape
        if output_shape is None:
            raise ValueError("A valid `output_shape` should be set in the init or initialize method.")
        self.values = torch.zeros(output_shape, dtype=self.dtype, device=self.device)
        self.counts = torch.zeros(output_shape, dtype=torch.uint8, device=self.device)
        self.is_initialized = True
        self.is_finalized = False

    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        if not self.is_initialized:
            raise ValueError("`AvgMerger` needs to be initialized before aggregation.")
        patch_size = values.shape[2:]
        map_slice = tuple(slice(loc, loc + size) for loc, size in zip(location, patch_size))
        map_slice = ensure_tuple_size(map_slice, values.ndim, pad_val=slice(None), pad_from_start=True)
        self.values[map_slice] += values
        self.counts[map_slice] += 1

    def finalize(self) -> torch.Tensor:
        """
        Finalize the merging by dividing values by counts.

        Note:
            To avoid creating a new tensor for the final results to save memory, after this method is called,
            `get_values()` method will return the "final" averaged values and not the accumulating values.
            Also calling `self.finalize` multiple times does not have any effect unless it is initialized again.

        Returns:
            torch.tensor: a tensor of merged patches
        """
        # guard against multiple call to finalize (before initialized again)
        if not self.is_finalized:
            # use in-place division to save space and create self.output alias
            self.output = self.values.div_(self.counts)
            # set finalize flag to protect performing in-place division again
            self.is_finalized = True
            # unset initialize flag to avoid further aggregation
            self.is_initialized = False

        return self.get_output()

    def get_values(self) -> torch.Tensor:
        """Get the accumulating values during aggregation or final averaged values after it is finalized.

        Note:
            - If called before finalizing, this method returns the accumulating values.
            - If called after finalizing, this method returns the final merged [and averaged] values.
            - Call self.finalize() to finalize.
        """
        return self.values

    def get_counts(self) -> torch.Tensor:
        """Get the aggregator tensor for number of samples."""
        return self.counts

    def get_output(self) -> torch.Tensor:
        """Get the output merged tensor."""
        return self.output
