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

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from inspect import _empty, isclass, signature
from typing import Any

import torch

from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import ToTensor
from monai.utils.misc import PathLike, ensure_tuple, ensure_tuple_rep

__all__ = ["Splitter", "SlidingWindowSplitter", "WSISlidingWindowSplitter"]


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
    """Split the input into patches with sliding window strategy and a possible overlap.
    It also allows to offset the starting position and filter the patches.

    Args:
        patch_size : the size of the patches to be generated.
        offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
        overlap: the amount of overlap between patches in each dimension. It can be either a float in
            the range of [0.0, 1.0) that defines relative overlap to the patch size, or it can be a non-negative int
            that defines number of pixels for overlap. Defaults to 0.0.
        filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
            return True for a patch to keep. Defaults to no filtering.
        pad_mode: string define the mode for `torch.nn.functional.pad`. The acceptable values are
            `"constant"`, `"reflect"`, `"replicate"`, `"circular"` or None. Default to `"constant"`.
            If None, no padding will be applied, so it will drop the patches crossing the border of
            the image (either when the offset is negative or the image is non-divisible by the patch_size).
        pad_value: the value for `"constant"` padding. Defaults to 0.
        non_spatial_ndim: number of non-spatial dimensions (e.g. batch, color)
        device: the device where the patches are generated. Defaults to the device of inputs.

    Note:
        When a scaler value is provided for `patch_size`, `offset`, or `overlap`,
            it is broadcasted to all the spatial dimensions.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int,
        overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
        offset: Sequence[int] | int = 0,
        filter_fn: Callable | None = None,
        pad_mode: str | None = None,
        pad_value: float | int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(patch_size=patch_size, device=device)
        self.offset = offset
        # check if fraction overlaps are within the range of [0, 1)
        if isinstance(ensure_tuple(overlap)[0], float) and any(ov < 0.0 or ov >= 1.0 for ov in ensure_tuple(overlap)):
            raise ValueError(
                f"Relative overlap must be between 0.0 and 1.0 but {overlap} is given. "
                "If you wish to use number of pixels as overlap, please provide integer numbers."
            )
        elif any(ov < 0 for ov in ensure_tuple(overlap)):
            raise ValueError(f"Number of pixels for overlap cannot be negative. {overlap} is given. ")

        self.overlap = overlap
        self.filter_fn = self._validate_filter_fn(filter_fn)
        # padding
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        # check a valid padding mode is provided if there is any negative offset.
        if not self.pad_mode and any(off < 0 for off in ensure_tuple(offset)):
            raise ValueError(f"Negative `offset`requires a valid padding mode but `mode` is set to {self.pad_mode}.")
        # batch and color are non-spatial dimensions
        self.non_spatial_ndim = 2

    @staticmethod
    def _validate_filter_fn(filter_fn):
        if callable(filter_fn):
            sig = signature(filter_fn)
            n_params = len(sig.parameters)
            num_pos_params = len([v for v in sig.parameters.values() if v.default is _empty])
            if n_params < 2:
                raise ValueError(
                    f"`filter_fn` requires to accept at least two parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {n_params} parameters."
                )
            elif num_pos_params > 2:
                raise ValueError(
                    f"`filter_fn` can have at most two positional parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {num_pos_params} positional parameters."
                )
        elif filter_fn is not None:
            raise ValueError(
                "`filter_fn` should be a callable with two input parameters (patch, location). "
                f"{type(filter_fn)} is given."
            )
        return filter_fn

    def _calculate_pad_size(self, spatial_shape, spatial_ndim, patch_size, offset, overlap):
        if not self.pad_mode:
            return [], False
        # initialize with zero
        pad_size = [0] * 2 * spatial_ndim
        # set the starting pad size only if the offset is negative
        pad_size[1::2] = (-min(off, 0) for off in offset)
        # set the ending pad size only if it is not divisible by the patch size
        pad_size[::2] = (
            0 if ps == 0 else (off - sh + ps) % round(ps - (ps * ov if isinstance(ov, float) else ov))
            for sh, off, ps, ov in zip(spatial_shape, offset, patch_size, overlap)
        )
        return pad_size, any(pad_size[1::2])

    def _get_valid_shape_parameters(
        self, spatial_shape: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[float, ...] | tuple[int, ...], tuple[int, ...]]:
        spatial_ndim = len(spatial_shape)
        # patch_size
        patch_size = ensure_tuple_rep(self.patch_size, spatial_ndim)
        # overlap
        overlap = ensure_tuple_rep(self.overlap, spatial_ndim)
        overlap = tuple(o if p else type(overlap[0])(0) for o, p in zip(overlap, patch_size))
        if any(ov > ps for ov, ps in zip(overlap, patch_size)):
            raise ValueError(f"`overlap` ({overlap}) cannot be larger than patch size ({patch_size}).")
        # offset
        offset = ensure_tuple_rep(self.offset, spatial_ndim)
        for off, ps, sh in zip(offset, patch_size, spatial_shape):
            if off < -ps:
                raise ValueError(f"Negative `offset` ({off}) cannot be larger than `patch_size` ({ps}) in magnitude.")
            if off >= sh:
                raise ValueError(f"`offset` ({off}) cannot be larger than inputs size ({sh}).")
        return patch_size, overlap, offset

    def _get_patch(self, inputs: Any, location: tuple[int, ...], patch_size: tuple[int, ...]) -> Any:
        slices = (slice(None),) * self.non_spatial_ndim + tuple(
            slice(loc, loc + ps) for loc, ps in zip(location, patch_size)
        )
        return inputs[slices]

    def __call__(self, inputs: Any) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: either a torch.Tensor with BCHW[D] dimensions, representing an image or a batch of images

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"The input should be a tensor. {type(inputs)} is given.")

        spatial_shape = inputs.shape[self.non_spatial_ndim :]
        spatial_ndim = len(spatial_shape)
        patch_size, overlap, offset = self._get_valid_shape_parameters(spatial_shape)
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        # Padding
        if self.pad_mode and any(pad_size):
            # pad the inputs
            inputs = torch.nn.functional.pad(inputs, pad_size[::-1], mode=self.pad_mode, value=self.pad_value)
            # update spatial shape
            spatial_shape = inputs.shape[self.non_spatial_ndim :]
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        # Splitting
        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            patch = self._get_patch(inputs, location, patch_size)  # type: ignore
            patch = ToTensor(device=self.device)(patch)  # type: ignore
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patch and yield
            if self.filter_fn is None:
                yield patch, location
            elif self.filter_fn(patch, location):
                yield patch, location


class WSISlidingWindowSplitter(SlidingWindowSplitter):
    """Split the whole slide image input into patches with sliding window strategy and a possible overlap.
    It also allows to offset the starting position and filter the patches.

    Args:
        patch_size : the size of the patches to be generated.
        offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
        overlap: the amount of overlap between patches in each dimension. It can be either a float in
            the range of [0.0, 1.0) that defines relative overlap to the patch size, or it can be a non-negative int
            that defines number of pixels for overlap. Defaults to 0.0.
        filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
            return True for a patch to keep. Defaults to no filtering.
        pad_mode: define the mode for padding. Either "constant" or None. Default to "constant".
            Depending on the reader
        device: the device where the patches are generated. Defaults to the device of inputs.
        reader: the module to be used for loading whole slide imaging. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
            - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        reader_kwargs: the arguments to pass to `WSIReader` or the provided whole slide reader class.
            For instance, level=2, dtype=torch.float32, etc.

    Note:
        When a scaler value is provided for `patch_size`, `offset`, or `overlap`,
        it is broadcasted to all the spatial dimensions.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int,
        overlap: Sequence[float] | float | Sequence[int] | int = 0.0,
        offset: Sequence[int] | int = 0,
        filter_fn: Callable | None = None,
        pad_mode: str | None = None,
        device: torch.device | str | None = None,
        reader: str | BaseWSIReader | type[BaseWSIReader] | None = None,
        **reader_kwargs: dict,
    ) -> None:
        if pad_mode and pad_mode != "constant":
            raise ValueError(
                f"The underlying wsi readers only support for constant padding. pad_mod={pad_mode} is given."
            )

        super().__init__(
            patch_size=patch_size, overlap=overlap, offset=offset, filter_fn=filter_fn, device=device, pad_mode=pad_mode
        )
        # Set WSI reader
        self._set_reader(reader, reader_kwargs)

    def _set_reader(self, reader: str | BaseWSIReader | type[BaseWSIReader] | None, reader_kwargs: dict) -> None:
        """
        Set the WSI reader object based on the input reader

        Args:
            reader: the module to be used for loading whole slide imaging. If `reader` is

                - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
                - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
                - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.
        """
        self.reader: WSIReader | BaseWSIReader
        self.reader_kwargs = reader_kwargs
        if isinstance(reader, str):
            self.reader = WSIReader(backend=reader, **self.reader_kwargs)
        elif isclass(reader) and issubclass(reader, BaseWSIReader):
            self.reader = reader(**self.reader_kwargs)
        elif isinstance(reader, BaseWSIReader):
            self.reader = reader
        else:
            raise ValueError(f"Unsupported reader type: {reader}.")

    def _get_patch(self, inputs: Any, location: tuple[int, ...], patch_size: tuple[int, ...]) -> Any:
        patch, _ = self.reader.get_data(wsi=inputs, location=location, size=patch_size)  # type: ignore
        return patch[None]

    def __call__(self, inputs: PathLike) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: the file path to a whole slide image.

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """
        if not isinstance(inputs, (str, os.PathLike)):
            raise ValueError(f"The input should be the path to the whole slide image. {type(inputs)} is given.")

        wsi = self.reader.read(inputs)
        spatial_shape: tuple = self.reader.get_size(wsi)
        spatial_ndim = len(spatial_shape)
        if spatial_ndim != 2:
            raise ValueError(f"WSIReader only support 2D images. {spatial_ndim} spatial dimension is provided.")
        patch_size, overlap, offset = self._get_valid_shape_parameters(spatial_shape)
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        # Padding (extend the spatial shape)
        if any(pad_size):
            spatial_shape = tuple(ss + ps for ss, ps in zip(spatial_shape, pad_size))
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        # Splitting (extracting patches)
        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            patch = self._get_patch(wsi, location, patch_size)
            patch = ToTensor(device=self.device)(patch)  # type: ignore
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patch and yield
            if self.filter_fn is None:
                yield patch, location
            elif self.filter_fn(patch, location):
                yield patch, location
