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
from typing import Any

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
    def initialize(self, inputs: torch.Tensor, in_patch: torch.Tensor, out_patch: torch.Tensor):
        """
        Initialize the merger.

        Args:
            inputs: a tensor of shape BCHW[D], representing a batch of input images
            in_patch: a tensor of shape BCH'W'[D'], representing a batch of input patches
            out_patch: a tensor of shape BC"H"W"[D"], representing a batch of input patches

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def aggregate(self, values: torch.Tensor, location: Sequence[int]):
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
    def finalize(self) -> Any:
        """
        Perform final operations for merging patches.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class AvgMerger(Merger):
    """Merge patches by taking average of the overlapping area

    Args:
        device: the device for aggregator tensors and final results.
        dtype: the dtype for aggregation and final result and .
    """

    def __init__(
        self, output_shape: tuple | None = None, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__(device, dtype)
        self.output_shape = output_shape

    def _get_device(self, in_patch, out_patch):
        """Define the device for value-aggregator/output tensor"""
        if isinstance(self.device, str):
            if self.device.lower() == "input":
                return in_patch.device
            if self.device.lower() == "output":
                return out_patch.device
        return self.device

    def _get_output_shape(self, inputs, in_patch, out_patch):
        """Define the shape for aggregator tensors"""
        if self.output_shape is None:
            in_spatial_shape = torch.tensor(inputs.shape[2:])
            in_patch_shape = torch.tensor(in_patch.shape[2:])
            out_patch_shape = torch.tensor(out_patch.shape[2:])
            batch_channel_shape = out_patch.shape[:2]
            spatial_shape = torch.round(in_spatial_shape * out_patch_shape / in_patch_shape).to(torch.int).tolist()
            return batch_channel_shape + tuple(spatial_shape)

        return self.output_shape

    def initialize(self, inputs: torch.Tensor, in_patch: torch.Tensor, out_patch: torch.Tensor):
        """
        Initialize the merger by creating tensors for aggregation (`values` and `counts`).

        Args:
            inputs: a tensor of shape BCHW[D], representing a batch of input images
            in_patch: a tensor of shape BCH'W'[D'], representing a batch of input patches
            out_patch: a tensor of shape BC"H"W"[D"], representing a batch of input patches
        """
        output_shape = self._get_output_shape(inputs, in_patch, out_patch)
        device = self._get_device(in_patch, out_patch)
        self.values = torch.zeros(output_shape, dtype=self.dtype, device=device)
        self.counts = torch.zeros(output_shape, dtype=torch.uint8, device=device)
        self.is_initialized = True
        self.is_finalized = False

    def aggregate(self, values: torch.Tensor, location: Sequence[int]):
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
            After calling this method, to avoid creating a new tensor for the final results to save memory,
            `self.get_values` returns the "final" averaged values and not the accumulating values.
            Also calling `self.finalize` multiple times does not have any effect unless it is initialized again.

        Returns:
            torch.tensor: a tensor of merged patches
        """
        if self.is_finalized:
            return self.values
        self.is_initialized = False
        self.is_finalized = True
        return self.values.div_(self.counts)

    def get_values(self) -> torch.Tensor:
        """Get the accumulating values during aggregation or final averaged values after it is finalized.

        Note:
            - If called before finalizing, this method returns the accumulating values.
            - If called after finalizing,  this method returns the final merged [and averaged] values.
            - Call self.finalize() for finalize.
        """
        return self.values

    def get_counts(self) -> torch.Tensor:
        """Get the aggregator tensor for number of samples."""
        return self.counts
