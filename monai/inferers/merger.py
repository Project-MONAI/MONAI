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
    Extend this class to support operations for `PatchInference`.
    There are two methods that must be implemented in the concrete classes:

        - aggregate: aggregate the values at their corresponding locations
        - finalize: perform any final process and return the merged output

    Args:
        output_shape: the shape of the merged output tensor. Default to None.
        device: the device where Merger tensors should reside.
    """

    def __init__(self, output_shape: Sequence[int] | None = None, device: torch.device | str | None = None) -> None:
        self.output_shape = output_shape
        self.device = device
        self.is_finalized = False

    @abstractmethod
    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """
        Aggregate values for merging.
        This method is being called in a loop and should add values to their corresponding location in the merged output results.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the output.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def finalize(self) -> Any:
        """
        Perform final operations for merging patches and return the final merged output.

        Returns:
            The results of merged patches, which is commonly a torch.Tensor representing the merged result, or
                a string representing the filepath to the merged results on disk.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class AvgMerger(Merger):
    """Merge patches by taking average of the overlapping area

    Args:
        output_shape: the shape of the merged output tensor.
        device: the device for aggregator tensors and final results.
        value_dtype: the dtype for value aggregating tensor and the final result.
        count_dtype: the dtype for sample counting tensor.
    """

    def __init__(
        self,
        output_shape: Sequence[int],
        device: torch.device | str = "cpu",
        value_dtype: torch.dtype = torch.float32,
        count_dtype: torch.dtype = torch.uint8,
    ) -> None:
        super().__init__(output_shape=output_shape, device=device)
        if not self.output_shape:
            raise ValueError(f"`output_shape` must be provided for `AvgMerger`. {self.output_shape} is give.")
        self.value_dtype = value_dtype
        self.count_dtype = count_dtype
        self.values = torch.zeros(self.output_shape, dtype=self.value_dtype, device=self.device)
        self.counts = torch.zeros(self.output_shape, dtype=self.count_dtype, device=self.device)

    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        if self.is_finalized:
            raise ValueError("`AvgMerger` is already finalized. Please instantiate a new object to aggregate.")
        patch_size = values.shape[2:]
        map_slice = tuple(slice(loc, loc + size) for loc, size in zip(location, patch_size))
        map_slice = ensure_tuple_size(map_slice, values.ndim, pad_val=slice(None), pad_from_start=True)
        self.values[map_slice] += values
        self.counts[map_slice] += 1

    def finalize(self) -> torch.Tensor:
        """
        Finalize merging by dividing values by counts and return the merged tensor.

        Notes:
            To avoid creating a new tensor for the final results (to save memory space),
            after this method is called, `get_values()` method will return the "final" averaged values,
            and not the accumulating values. Also calling `finalize()` multiple times does not have any effect.

        Returns:
            torch.tensor: a tensor of merged patches
        """
        # guard against multiple call to finalize
        if not self.is_finalized:
            # use in-place division to save space
            self.values.div_(self.counts)
            # set finalize flag to protect performing in-place division again
            self.is_finalized = True

        return self.values

    def get_values(self) -> torch.Tensor:
        """
        Get the accumulated values during aggregation or final averaged values after it is finalized.

        Returns:
            Merged (averaged) output tensor.

        Notes:
            - If called before calling `finalize()`, this method returns the accumulating values.
            - If called after calling `finalize()`, this method returns the final merged [and averaged] values.
        """
        return self.values

    def get_counts(self) -> torch.Tensor:
        """
        Get the aggregator tensor for number of samples.

        Returns:
            torch.Tensor: Number of accumulated samples at each location.
        """
        return self.counts
