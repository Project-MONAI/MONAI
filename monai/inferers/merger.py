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
from typing import Sequence, Any

import torch
from monai.utils.enums import PatchKeys
from monai.utils import ensure_tuple_size

__all__ = ["Merger"]


class Merger(ABC):
    """
    A base class for merging patches.
    Extend this class to support operations for `PatchInference`, e.g. `AvgMerger`.

    Args:
        device: the device where Merger tensors should reside.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        self.device = device

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
        device: the device for `values` and `count` tensor aggregator.
        dtype: the dtype of `values` tensor aggregator.
    """

    def __init__(self, device: torch.device | str | None = None, dtype=torch.float32) -> None:
        super().__init__(device)
        self.dtype = dtype

    def initialize(self, inputs: torch.Tensor, in_patch: torch.Tensor, out_patch: torch.Tensor):
        """
        Initialize the merger by creating tensors for aggregation (`values` and `counts`).

        Args:
            inputs: a tensor of shape BCHW[D], representing a batch of input images
            in_patch: a tensor of shape BCH'W'[D'], representing a batch of input patches
            out_patch: a tensor of shape BC"H"W"[D"], representing a batch of input patches
        """
        super().__init__()
        in_spatial_shape = torch.tensor(inputs.shape[2:])
        in_patch_shape = torch.tensor(in_patch.shape[2:])
        out_patch_shape = torch.tensor(out_patch.shape[2:])
        batch_channel_shape = out_patch.shape[:2]
        spatial_shape = torch.round(in_spatial_shape * out_patch_shape / in_patch_shape).to(torch.int).tolist()
        output_shape = batch_channel_shape + tuple(spatial_shape)
        # decide on the device for aggregators
        device: torch.device | str
        if self.device is None:
            device = inputs.device
        elif isinstance(self.device, str):
            if self.device.lower() == "input":
                device = inputs.device
            elif self.device.lower() == "output":
                device = out_patch.device
            else:
                device = self.device
        else:
            device = self.device
        # initialize values and counts tensors for aggregation
        self.values = torch.zeros(output_shape, dtype=self.dtype, device=device)
        self.counts = torch.zeros(output_shape, dtype=torch.uint8, device=device)

    def aggregate(self, values: torch.Tensor, location: Sequence[int]):
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        patch_size = values.shape[2:]
        map_slice = tuple(slice(loc, loc + size) for loc, size in zip(location.tolist(), patch_size))
        map_slice = ensure_tuple_size(map_slice, values.ndim, pad_val=slice(None), pad_from_start=True)
        self.values[map_slice] += values
        self.counts[map_slice] += 1

    def finalize(self) -> torch.Tensor:
        """
        Finalize the merging by dividing values by counts.
        Note that after finalize, `values` tensor represents the average and not the aggregated values.
        This avoid creating a new tensor for the final results and save memory.

        Returns:
            torch.tensor: a tensor of merged patches
        """
        return self.values.div_(self.counts)

    def get_values(self):
        return self.values

    def get_counts(self):
        return self.counts
