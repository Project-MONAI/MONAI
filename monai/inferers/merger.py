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

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from monai.utils import ensure_tuple_size, optional_import, require_pkg

if TYPE_CHECKING:
    import zarr
else:
    zarr, _ = optional_import("zarr")

__all__ = ["Merger", "AvgMerger", "ZarrAvgMerger"]


class Merger(ABC):
    """
    A base class for merging patches.
    Extend this class to support operations for `PatchInference`.
    There are two methods that must be implemented in the concrete classes:

        - aggregate: aggregate the values at their corresponding locations
        - finalize: perform any final process and return the merged output

    Args:
        merged_shape: the shape of the tensor required to merge the patches.
        cropped_shape: the shape of the final merged output tensor.
            If not provided, it will be the same as `merged_shape`.
        device: the device where Merger tensors should reside.
    """

    def __init__(
        self,
        merged_shape: Sequence[int],
        cropped_shape: Sequence[int] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.merged_shape = merged_shape
        self.cropped_shape = self.merged_shape if cropped_shape is None else cropped_shape
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
        merged_shape: the shape of the tensor required to merge the patches.
        cropped_shape: the shape of the final merged output tensor.
            If not provided, it will be the same as `merged_shape`.
        device: the device for aggregator tensors and final results.
        value_dtype: the dtype for value aggregating tensor and the final result.
        count_dtype: the dtype for sample counting tensor.
    """

    def __init__(
        self,
        merged_shape: Sequence[int],
        cropped_shape: Sequence[int] | None = None,
        value_dtype: torch.dtype = torch.float32,
        count_dtype: torch.dtype = torch.uint8,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(merged_shape=merged_shape, cropped_shape=cropped_shape, device=device)
        if not self.merged_shape:
            raise ValueError(f"`merged_shape` must be provided for `AvgMerger`. {self.merged_shape} is give.")
        self.value_dtype = value_dtype
        self.count_dtype = count_dtype
        self.values = torch.zeros(self.merged_shape, dtype=self.value_dtype, device=self.device)
        self.counts = torch.zeros(self.merged_shape, dtype=self.count_dtype, device=self.device)

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
            # finalize the shape
            self.values = self.values[tuple(slice(0, end) for end in self.cropped_shape)]
            # set finalize flag to protect performing in-place division again
            self.is_finalized = True

        return self.values

    def get_output(self) -> torch.Tensor:
        """
        Get the final merged output.

        Returns:
            torch.Tensor: merged output.
        """
        return self.finalize()

    def get_values(self) -> torch.Tensor:
        """
        Get the accumulated values during aggregation or final averaged values after it is finalized.

        Returns:
            torch.tensor: aggregated values.

        Notes:
            - If called before calling `finalize()`, this method returns the accumulating values.
            - If called after calling `finalize()`, this method returns the final merged [and averaged] values.
        """
        return self.values

    def get_counts(self) -> torch.Tensor:
        """
        Get the aggregator tensor for number of samples.

        Returns:
            torch.Tensor: number of accumulated samples at each location.
        """
        return self.counts


@require_pkg(pkg_name="zarr")
class ZarrAvgMerger(Merger):
    """Merge patches by taking average of the overlapping area and store the results in zarr array.

    Zarr is a format for the storage of chunked, compressed, N-dimensional arrays.
    Zarr data can be stored in any storage system that can be represented as a key-value store,
    like POSIX file systems, cloud object storage, zip files, and relational and document databases.
    See https://zarr.readthedocs.io/en/stable/ for more details.
    It is particularly useful for storing N-dimensional arrays too large to fit into memory.
    One specific use case of this class is to merge patches extracted from whole slide images (WSI),
    where the merged results do not fit into memory and need to be stored on a file system.

    Args:
        merged_shape: the shape of the tensor required to merge the patches.
        cropped_shape: the shape of the final merged output tensor.
            If not provided, it will be the same as `merged_shape`.
        dtype: the dtype for the final merged result. Default is `float32`.
        value_dtype: the dtype for value aggregating tensor and the final result. Default is `float32`.
        count_dtype: the dtype for sample counting tensor. Default is `uint8`.
        store: the zarr store to save the final results. Default is "merged.zarr".
        value_store: the zarr store to save the value aggregating tensor. Default is a temporary store.
        count_store: the zarr store to save the sample counting tensor. Default is a temporary store.
        compressor: the compressor for final merged zarr array. Default is "default".
        value_compressor: the compressor for value aggregating zarr array. Default is None.
        count_compressor: the compressor for sample counting zarr array. Default is None.
        chunks : int or tuple of ints that defines the chunk shape, or boolean. Default is True.
            If True, chunk shape will be guessed from `shape` and `dtype`.
            If False, it will be set to `shape`, i.e., single chunk for the whole array.
            If an int, the chunk size in each dimension will be given by the value of `chunks`.
    """

    def __init__(
        self,
        merged_shape: Sequence[int],
        cropped_shape: Sequence[int] | None = None,
        dtype: np.dtype | str = "float32",
        value_dtype: np.dtype | str = "float32",
        count_dtype: np.dtype | str = "uint8",
        store: zarr.storage.Store | str = "merged.zarr",
        value_store: zarr.storage.Store | str | None = None,
        count_store: zarr.storage.Store | str | None = None,
        compressor: str = "default",
        value_compressor: str | None = None,
        count_compressor: str | None = None,
        chunks: Sequence[int] | bool = True,
        thread_locking: bool = True,
    ) -> None:
        super().__init__(merged_shape=merged_shape, cropped_shape=cropped_shape)
        if not self.merged_shape:
            raise ValueError(f"`merged_shape` must be provided for `ZarrAvgMerger`. {self.merged_shape} is give.")
        self.output_dtype = dtype
        self.value_dtype = value_dtype
        self.count_dtype = count_dtype
        self.store = store
        self.value_store = zarr.storage.TempStore() if value_store is None else value_store
        self.count_store = zarr.storage.TempStore() if count_store is None else count_store
        self.chunks = chunks
        self.compressor = compressor
        self.value_compressor = value_compressor
        self.count_compressor = count_compressor
        self.output = zarr.empty(
            shape=self.merged_shape,
            chunks=self.chunks,
            dtype=self.output_dtype,
            compressor=self.compressor,
            store=self.store,
            overwrite=True,
        )
        self.values = zarr.zeros(
            shape=self.merged_shape,
            chunks=self.chunks,
            dtype=self.value_dtype,
            compressor=self.value_compressor,
            store=self.value_store,
            overwrite=True,
        )
        self.counts = zarr.zeros(
            shape=self.merged_shape,
            chunks=self.chunks,
            dtype=self.count_dtype,
            compressor=self.count_compressor,
            store=self.count_store,
            overwrite=True,
        )
        self.lock: threading.Lock | nullcontext
        if thread_locking:
            # use lock to protect the in-place addition during aggregation
            self.lock = threading.Lock()
        else:
            # use nullcontext to avoid the locking if not needed
            self.lock = nullcontext()

    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """
        Aggregate values for merging.

        Args:
            values: a tensor of shape BCHW[D], representing the values of inference output.
            location: a tuple/list giving the top left location of the patch in the original image.
        """
        if self.is_finalized:
            raise ValueError("`ZarrAvgMerger` is already finalized. Please instantiate a new object to aggregate.")
        patch_size = values.shape[2:]
        map_slice = tuple(slice(loc, loc + size) for loc, size in zip(location, patch_size))
        map_slice = ensure_tuple_size(map_slice, values.ndim, pad_val=slice(None), pad_from_start=True)
        with self.lock:
            self.values[map_slice] += values.numpy()
            self.counts[map_slice] += 1

    def finalize(self) -> zarr.Array:
        """
        Finalize merging by dividing values by counts and return the merged tensor.

        Notes:
            To avoid creating a new tensor for the final results (to save memory space),
            after this method is called, `get_values()` method will return the "final" averaged values,
            and not the accumulating values. Also calling `finalize()` multiple times does not have any effect.

        Returns:
            zarr.Array: a zarr array of of merged patches
        """
        # guard against multiple calls to finalize
        if not self.is_finalized:
            # use chunks for division to fit into memory
            for chunk in iterate_over_chunks(self.values.chunks, self.values.cdata_shape):
                self.output[chunk] = self.values[chunk] / self.counts[chunk]
            # finalize the shape
            self.output.resize(self.cropped_shape)
            # set finalize flag to protect performing in-place division again
            self.is_finalized = True

        return self.output

    def get_output(self) -> zarr.Array:
        """
        Get the final merged output.

        Returns:
            zarr.Array: Merged (averaged) output tensor.
        """
        return self.output

    def get_values(self) -> zarr.Array:
        """
        Get the accumulated values during aggregation

        Returns:
            zarr.Array: aggregated values.

        """
        return self.values

    def get_counts(self) -> zarr.Array:
        """
        Get the aggregator tensor for number of samples.

        Returns:
            zarr.Array: Number of accumulated samples at each location.
        """
        return self.counts


def iterate_over_chunks(chunks, cdata_shape, slice_tuple=()):
    """
    Iterate over chunks of a given shape.

    Args:
        chunks: the chunk shape
        cdata_shape: the shape of the data in chunks
        slice_tuple: the slice tuple to be used for indexing

    Raises:
        ValueError: When the length of chunks and cdata_shape are not the same.

    Yields:
        slices of the data
    """
    if len(chunks) != len(cdata_shape):
        raise ValueError("chunks and cdata_shape must have the same length")
    if len(chunks) == 1:
        for i in range(cdata_shape[0]):
            yield slice_tuple + (slice(i * chunks[0], (i + 1) * chunks[0]),)
    else:
        for i in range(cdata_shape[0]):
            yield from iterate_over_chunks(
                chunks[1:], cdata_shape[1:], slice_tuple + (slice(i * chunks[0], (i + 1) * chunks[0]),)
            )
