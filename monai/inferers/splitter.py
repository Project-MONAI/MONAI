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
from collections.abc import Callable, Iterable, Sequence
from inspect import _empty, signature
from typing import Any

import torch

from monai.data.utils import iter_patch_position
from monai.utils.enums import PytorchPadMode
from monai.utils.misc import ensure_tuple, ensure_tuple_rep
from monai.utils.module import look_up_option

__all__ = ["Splitter", "SlidingWindowSplitter"]


class Splitter(ABC):
    """
    A base class for splitting the inputs into iterable tuple of patches and locations
    Extend this class to support operations for `PatchInference`, e.g. SlidingPatchSplitter.

    Args:
        patch_size: the size of patches to be generated.
        device: the device where the patches are generated.
    """

    def __init__(self, patch_size: Sequence[int] | int, device: torch.device | str | None = None) -> None:
        self.patch_size = patch_size
        self.device = device

    @abstractmethod
    def __call__(self, inputs: Any) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """
        Split the input image (or batch of images) into patches and return pairs of (patch, location).
        Where location is the coordinate of top left [front] corner of a patch.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class SlidingWindowSplitter(Splitter):
    def __init__(
        self,
        patch_size: Sequence[int] | int,
        offset: Sequence[int] | int = 0,
        overlap: Sequence[float] | float = 0.0,
        filter_fn: Callable | None = None,
        device: torch.device | str | None = None,
        pad_mode: str | None = PytorchPadMode.CONSTANT,
        **pad_kwargs: Any,
    ) -> None:
        """Split the input into patches with sliding window strategy and a possible overlap.
        If also allows to offset the starting position and filter the patches.

        Args:
            patch_size : the size of the patch to be generated.
            offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
            overlap: the amount of overlap between patches in each dimension [0, 1). Defaults to 0.0.
            filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
                return True for a patch to keep. Defaults to no filtering.
            device: the device where the patches are generated. Defaults to the device of inputs.
            pad_mode: the pad mode when the patches are extracted from outside of the image
                (either when the offset is negative or the image is non-divisible by the patch_size).
                If set to `None`, the last incomplete patch will be dropped. Defaults to PytorchPadMode.CONSTANT.
            pad_kwargs: other arguments for `torch.nn.functional.pad`.

        Note:
            If only one scaler value is provided for `patch_size`, `offset`, or `overlap`,
                it will be broadcasted to all the spatial dimensions.
        """
        super().__init__(patch_size=patch_size, device=device)
        self.offset = offset
        if any(ov < 0 or ov >= 1 for ov in ensure_tuple(overlap)):
            raise ValueError(f"Overlap must be between 0 and 1 but {overlap} is given.")
        self.overlap = overlap
        self.filter_fn = self._get_valid_filter_fn(filter_fn)
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs

    def _get_valid_filter_fn(self, filter_fn):
        if filter_fn is None:
            return
        if callable(filter_fn):
            sig = signature(filter_fn)
            n_params = len(sig.parameters)
            n_pos_params = len([v for v in sig.parameters.values() if v.default is _empty])
            if n_params < 2:
                raise ValueError(
                    f"`patch_filter_fn` requires to accept at least two parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {n_params} parameters."
                )
            elif n_pos_params > 2:
                raise ValueError(
                    f"`patch_filter_fn` can have at most two positional parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {n_pos_params} positional parameters."
                )
            return filter_fn
        raise ValueError(
            "`patch_filter_fn` should be a callable with two input parameters (patch, location). "
            f"{type(filter_fn)} is given."
        )

    def _get_valid_patch_size(self, spatial_ndim):
        return ensure_tuple_rep(self.patch_size, spatial_ndim)

    def _get_valid_overlap(self, spatial_ndim, patch_size):
        # broadcast overlap is possible
        overlap = ensure_tuple_rep(self.overlap, spatial_ndim)
        # keep overlap only in patching dimensions
        return tuple(o if p else 0.0 for o, p in zip(overlap, patch_size))

    def _get_valid_offset(self, spatial_shape, spatial_ndim, patch_size):
        offset = ensure_tuple_rep(self.offset, spatial_ndim)
        for off, ps, ins in zip(offset, patch_size, spatial_shape):
            if off < 0 and not self.pad_mode:
                raise ValueError(
                    f"Negative `offset` ({off}) requires a valid padding mode, "
                    f"but `pad_mod` is set to {self.pad_mode}."
                )
            if off < -ps:
                raise ValueError(f"Negative `offset` ({off}) cannot be larger than `patch_size` ({ps}) in magnitude.")
            if off >= ins:
                raise ValueError(f"`offset` ({off}) cannot be larger than inputs size ({ins}).")
        return offset

    def _calculate_pad_size(self, spatial_shape, spatial_ndim, patch_size, offset, overlap):
        if not self.pad_mode:
            return [], False
        # initialize with zero
        pad_size = [0] * 2 * spatial_ndim
        # set the starting pad size only if the offset is negative
        pad_size[1::2] = (-min(off, 0) for off in offset)
        # set the ending pad size only if it is not divisible by the patch size
        pad_size[::2] = (
            0 if ps == 0 else (off - ins + ps) % round(ps * (1.0 - ov))
            for ins, off, ps, ov in zip(spatial_shape, offset, patch_size, overlap)
        )
        return pad_size, any(pad_size[1::2])

    def __call__(self, inputs: torch.Tensor) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: a torch.Tensor with BCHW[D] dimensions, representing an image or a batch of images.

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """
        n_non_spatial_dims = 2
        spatial_ndim = inputs.ndim - n_non_spatial_dims
        spatial_shape = inputs.shape[n_non_spatial_dims:]
        patch_size = self._get_valid_patch_size(spatial_ndim)
        overlap = self._get_valid_overlap(spatial_ndim, patch_size)
        offset = self._get_valid_offset(spatial_shape, spatial_ndim, patch_size)
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        if any(pad_size):
            # pad the inputs
            inputs = torch.nn.functional.pad(
                inputs, pad_size[::-1], look_up_option(self.pad_mode, PytorchPadMode).value, **self.pad_kwargs
            )
            # update spatial shape
            spatial_shape = inputs.shape[n_non_spatial_dims:]
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            slices = (slice(None),) * 2 + tuple(slice(loc, loc + ps) for loc, ps in zip(location, patch_size))
            patch = inputs[slices]
            # send the patch to target device
            if self.device:
                patch.to(self.device)
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patches and yield
            if self.filter_fn is None:
                yield patch, location
            elif self.filter_fn(patch, location):
                yield patch, location
