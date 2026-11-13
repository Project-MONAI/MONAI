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
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from monai.utils import (
    deprecated_arg,
    ensure_tuple_size,
    get_package_version,
    optional_import,
    require_pkg,
    version_geq,
)

if TYPE_CHECKING:
    import zarr
else:
    zarr, _ = optional_import("zarr")

__all__ = ["Merger", "AvgMerger", "StreamingSlabAvgMerger", "ZarrAvgMerger"]


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
        if merged_shape is None:
            raise ValueError("Argument `merged_shape` must be provided")

        self.merged_shape: tuple[int, ...] = tuple(merged_shape)
        self.cropped_shape: tuple[int, ...] = tuple(self.merged_shape if cropped_shape is None else cropped_shape)
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


class StreamingSlabAvgMerger(Merger):
    """Merge patches by maintaining an active slab in memory, streaming finalized slices
    into a destination tensor along ``slab_dim``.

    Args:
        merged_shape: the shape of the tensor required to merge the patches (BCHW[D]).
        roi_size: the spatial window size of patches.
        slab_dim: the spatial dimension along which streaming slab progresses (default 0).
        cropped_shape: the shape of the final merged output tensor.
        device: device for destination merged tensor (default "cpu").
        slab_device: device for active slab accumulation (e.g. "cuda" or sw_device).
        value_dtype: the dtype for values. Default torch.float32.
        count_dtype: the dtype for counts. Default torch.float32.
    """

    def __init__(
        self,
        merged_shape: Sequence[int],
        roi_size: Sequence[int],
        slab_dim: int = 0,
        cropped_shape: Sequence[int] | None = None,
        device: torch.device | str = "cpu",
        slab_device: torch.device | str | None = None,
        value_dtype: torch.dtype = torch.float32,
        count_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(merged_shape=merged_shape, cropped_shape=cropped_shape, device=device)
        self.roi_size = tuple(roi_size)
        num_spatial_dims = len(self.merged_shape) - 2
        if slab_dim < -num_spatial_dims or slab_dim >= num_spatial_dims:
            raise ValueError(f"slab_dim must be in [{-num_spatial_dims}, {num_spatial_dims - 1}], got {slab_dim}.")
        if slab_dim < 0:
            slab_dim += num_spatial_dims
        self.slab_dim = slab_dim
        self.slab_device = slab_device or device
        self.value_dtype = value_dtype
        self.count_dtype = count_dtype

        self.output = torch.full(self.merged_shape, float("nan"), dtype=self.value_dtype, device=self.device)
        self.P_slab = self.roi_size[self.slab_dim]
        slab_shape = list(self.merged_shape)
        slab_shape[self.slab_dim + 2] = self.P_slab
        self.active_values = torch.zeros(slab_shape, dtype=self.value_dtype, device=self.slab_device)
        count_shape = [1, 1] + slab_shape[2:]
        self.active_counts = torch.zeros(count_shape, dtype=self.count_dtype, device=self.slab_device)
        self.curr_z_start: int | None = None
        self.num_spatial_dims = num_spatial_dims

    def _retire_slices(self, count: int, z_start: int) -> None:
        """Normalize finalized active slab slices and stream them to destination output.

        Args:
            count: number of slices along ``slab_dim`` to retire.
            z_start: starting coordinate along ``slab_dim`` in the destination tensor.
        """
        count = min(count, self.P_slab)
        if count <= 0:
            return
        retire_val_slice = [slice(None), slice(None)] + [slice(None)] * self.num_spatial_dims
        retire_val_slice[self.slab_dim + 2] = slice(0, count)
        retire_cnt_slice = [slice(None), slice(None)] + [slice(None)] * self.num_spatial_dims
        retire_cnt_slice[self.slab_dim + 2] = slice(0, count)

        dst_slice = [slice(None), slice(None)] + [slice(None)] * self.num_spatial_dims
        dst_slice[self.slab_dim + 2] = slice(z_start, z_start + count)

        retire_val = self.active_values[tuple(retire_val_slice)] / self.active_counts[tuple(retire_cnt_slice)]
        self.output[tuple(dst_slice)] = retire_val.to(device=self.device)

        rem = self.P_slab - count
        if rem > 0:
            src_shift = [slice(None)] * (self.num_spatial_dims + 2)
            src_shift[self.slab_dim + 2] = slice(count, count + rem)
            dst_shift = [slice(None)] * (self.num_spatial_dims + 2)
            dst_shift[self.slab_dim + 2] = slice(0, rem)
            clear_shift = [slice(None)] * (self.num_spatial_dims + 2)
            clear_shift[self.slab_dim + 2] = slice(rem, self.P_slab)

            self.active_values[tuple(dst_shift)] = self.active_values[tuple(src_shift)].clone()
            self.active_values[tuple(clear_shift)].zero_()
            self.active_counts[tuple(dst_shift)] = self.active_counts[tuple(src_shift)].clone()
            self.active_counts[tuple(clear_shift)].zero_()
        else:
            self.active_values.zero_()
            self.active_counts.zero_()

    def aggregate(self, values: torch.Tensor, location: Sequence[int]) -> None:
        """Aggregate a patch prediction into the active slab, retiring completed slices.

        Args:
            values: patch prediction tensor of shape BCHW[D].
            location: starting coordinate of the patch in the output tensor.

        Raises:
            ValueError: if merger is already finalized, or if patches arrive out-of-order
                with decreasing coordinates along ``slab_dim``.
        """
        if self.is_finalized:
            raise ValueError("`StreamingSlabAvgMerger` is already finalized.")
        loc_slab = int(location[self.slab_dim])
        if self.curr_z_start is None:
            self.curr_z_start = loc_slab

        if loc_slab < self.curr_z_start:
            raise ValueError(
                f"`StreamingSlabAvgMerger` requires patches to arrive in non-decreasing order along "
                f"`slab_dim` ({self.slab_dim}). Got patch location {loc_slab} < current slab start {self.curr_z_start}."
            )

        if loc_slab > self.curr_z_start:
            num_retire = loc_slab - self.curr_z_start
            self._retire_slices(min(num_retire, self.P_slab), self.curr_z_start)
            self.curr_z_start = loc_slab

        patch_size = values.shape[2:]
        local_val_s = [slice(None), slice(None)] + [slice(None)] * self.num_spatial_dims
        local_cnt_s = [slice(None), slice(None)] + [slice(None)] * self.num_spatial_dims
        for d in range(self.num_spatial_dims):
            if d == self.slab_dim:
                offset = loc_slab - self.curr_z_start
                sz = patch_size[d]
                local_val_s[d + 2] = slice(offset, offset + sz)
                local_cnt_s[d + 2] = slice(offset, offset + sz)
            else:
                loc = location[d]
                sz = patch_size[d]
                local_val_s[d + 2] = slice(loc, loc + sz)
                local_cnt_s[d + 2] = slice(loc, loc + sz)

        self.active_values[tuple(local_val_s)] += values.to(device=self.slab_device)
        self.active_counts[tuple(local_cnt_s)] += 1

    def finalize(self) -> torch.Tensor:
        """Finalize and return the merged tensor.

        Returns:
            torch.Tensor: merged tensor on ``device``, with cropped spatial dimensions if specified.
        """
        if not self.is_finalized:
            if self.curr_z_start is not None:
                total_slab_size = self.merged_shape[self.slab_dim + 2]
                remaining_retire = total_slab_size - self.curr_z_start
                self._retire_slices(min(self.P_slab, remaining_retire), self.curr_z_start)
            if self.cropped_shape != self.merged_shape:
                self.output = self.output[tuple(slice(0, end) for end in self.cropped_shape)]
            self.is_finalized = True
        return self.output

    def get_output(self) -> torch.Tensor:
        """Get the final merged output tensor.

        Returns:
            torch.Tensor: merged output.
        """
        return self.finalize()


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
        compressor: the compressor for final merged zarr array. Default is None.
            Deprecated since 1.5.0 and will be removed in 1.7.0. Use codecs instead.
        value_compressor: the compressor for value aggregating zarr array. Default is None.
            Deprecated since 1.5.0 and will be removed in 1.7.0. Use value_codecs instead.
        count_compressor: the compressor for sample counting zarr array. Default is None.
            Deprecated since 1.5.0 and will be removed in 1.7.0. Use count_codecs instead.
        codecs: the codecs for final merged zarr array. Default is None.
            For zarr v3, this is a list of codec configurations. See zarr documentation for details.
        value_codecs: the codecs for value aggregating zarr array. Default is None.
            For zarr v3, this is a list of codec configurations. See zarr documentation for details.
        count_codecs: the codecs for sample counting zarr array. Default is None.
            For zarr v3, this is a list of codec configurations. See zarr documentation for details.
        chunks : int or tuple of ints that defines the chunk shape, or boolean. Default is True.
            If True, chunk shape will be guessed from `shape` and `dtype`.
            If False, it will be set to `shape`, i.e., single chunk for the whole array.
            If an int, the chunk size in each dimension will be given by the value of `chunks`.
        thread_locking: whether to use thread locking during aggregation. Default is True.
    """

    @deprecated_arg(
        name="compressor", since="1.5.0", removed="1.7.0", new_name="codecs", msg_suffix="Please use 'codecs' instead."
    )
    @deprecated_arg(
        name="value_compressor",
        since="1.5.0",
        removed="1.7.0",
        new_name="value_codecs",
        msg_suffix="Please use 'value_codecs' instead.",
    )
    @deprecated_arg(
        name="count_compressor",
        since="1.5.0",
        removed="1.7.0",
        new_name="count_codecs",
        msg_suffix="Please use 'count_codecs' instead.",
    )
    def __init__(
        self,
        merged_shape: Sequence[int],
        cropped_shape: Sequence[int] | None = None,
        dtype: np.dtype | str = "float32",
        value_dtype: np.dtype | str = "float32",
        count_dtype: np.dtype | str = "uint8",
        store: zarr.storage.Store | str = "merged.zarr",  # type: ignore
        value_store: zarr.storage.Store | str | None = None,  # type: ignore
        count_store: zarr.storage.Store | str | None = None,  # type: ignore
        compressor: str | None = None,
        value_compressor: str | None = None,
        count_compressor: str | None = None,
        codecs: list | None = None,
        value_codecs: list | None = None,
        count_codecs: list | None = None,
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
        # Use separate tmpdir attributes so both TemporaryDirectory objects remain alive
        # for the full lifetime of this instance. Previously a single `self.tmpdir` was
        # overwritten when both value_store and count_store were None, causing the first
        # TemporaryDirectory to be garbage-collected immediately.
        self.value_tmpdir: TemporaryDirectory | None = None
        self.count_tmpdir: TemporaryDirectory | None = None

        # Handle zarr v3 vs older versions
        is_zarr_v3 = version_geq(get_package_version("zarr"), "3.0.0")

        if is_zarr_v3:
            if value_store is None:
                self.value_tmpdir = TemporaryDirectory()
                self.value_store = zarr.storage.LocalStore(self.value_tmpdir.name)  # type: ignore
            else:
                self.value_store = value_store  # type: ignore
            if count_store is None:
                self.count_tmpdir = TemporaryDirectory()
                self.count_store = zarr.storage.LocalStore(self.count_tmpdir.name)  # type: ignore
            else:
                self.count_store = count_store  # type: ignore
        else:
            self.value_store = zarr.storage.TempStore() if value_store is None else value_store  # type: ignore
            self.count_store = zarr.storage.TempStore() if count_store is None else count_store  # type: ignore

        # zarr 3.3 rejects boolean chunk arguments; `None` there means auto-chunking,
        # which is what `chunks=True` meant before. On zarr < 3.3 `None` instead means
        # a single whole-shape chunk, so the substitution must not be applied there.
        if chunks is True and version_geq(get_package_version("zarr"), "3.3.0"):
            self.chunks: Sequence[int] | bool | None = None
        else:
            self.chunks = chunks

        # Initialize codecs/compressor attributes with proper types
        self.codecs: list | None = None
        self.value_codecs: list | None = None
        self.count_codecs: list | None = None

        if is_zarr_v3:
            # For zarr v3, use codecs or convert compressor to codecs
            if codecs is not None:
                self.codecs = codecs
            elif compressor is not None:
                if isinstance(compressor, (list, tuple)):
                    self.codecs = compressor
                else:
                    self.codecs = [compressor]
            else:
                self.codecs = None

            if value_codecs is not None:
                self.value_codecs = value_codecs
            elif value_compressor is not None:
                if isinstance(value_compressor, (list, tuple)):
                    self.value_codecs = value_compressor
                else:
                    self.value_codecs = [value_compressor]
            else:
                self.value_codecs = None

            if count_codecs is not None:
                self.count_codecs = count_codecs
            elif count_compressor is not None:
                if isinstance(count_compressor, (list, tuple)):
                    self.count_codecs = count_compressor
                else:
                    self.count_codecs = [count_compressor]
            else:
                self.count_codecs = None
        else:
            # For zarr v2, use compressors
            if isinstance(codecs, (list, tuple)):
                self.codecs = codecs[0] if codecs else None
            elif codecs is not None:
                self.codecs = codecs
            else:
                self.codecs = compressor  # type: ignore[assignment]
            if isinstance(value_codecs, (list, tuple)):
                self.value_codecs = value_codecs[0] if value_codecs else None
            elif value_codecs is not None:
                self.value_codecs = value_codecs
            else:
                self.value_codecs = value_compressor  # type: ignore[assignment]
            if isinstance(count_codecs, (list, tuple)):
                self.count_codecs = count_codecs[0] if count_codecs else None
            elif count_codecs is not None:
                self.count_codecs = count_codecs
            else:
                self.count_codecs = count_compressor  # type: ignore[assignment]

        # Create zarr arrays with appropriate parameters based on version
        if is_zarr_v3:
            self.output = zarr.empty(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.output_dtype,
                codecs=self.codecs,
                store=self.store,
                overwrite=True,
            )
            self.values = zarr.zeros(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.value_dtype,
                codecs=self.value_codecs,
                store=self.value_store,
                overwrite=True,
            )
            self.counts = zarr.zeros(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.count_dtype,
                codecs=self.count_codecs,
                store=self.count_store,
                overwrite=True,
            )
        else:
            self.output = zarr.empty(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.output_dtype,
                compressor=self.codecs,
                store=self.store,
                overwrite=True,
            )
            self.values = zarr.zeros(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.value_dtype,
                compressor=self.value_codecs,
                store=self.value_store,
                overwrite=True,
            )
            self.counts = zarr.zeros(
                shape=self.merged_shape,
                chunks=self.chunks,
                dtype=self.count_dtype,
                compressor=self.count_codecs,
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
            self.counts[map_slice] += 1  # type: ignore[operator]

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
                self.output[chunk] = self.values[chunk] / self.counts[chunk]  # type: ignore[operator]
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
