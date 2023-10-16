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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from inspect import _empty, isclass, signature
from typing import Any

import torch

from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms.utility.array import convert_to_tensor
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
    def get_input_shape(self, inputs: Any) -> tuple:
        """
        Return the input spatial shape.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_padded_shape(self, inputs: Any) -> tuple:
        """
        Return the actual spatial shape covered by the output split patches.
        For instance, if the input image is padded, the actual spatial shape will be enlarged
        and not the same as input spatial shape.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

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
    """
    Splits the input into patches with sliding window strategy and a possible overlap.
    It also allows offsetting the starting position and filtering the patches.

    Args:
        patch_size : the size of the patches to be generated.
        offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
        overlap: the amount of overlap between patches in each dimension. It can be either a float in
            the range of [0.0, 1.0) that defines relative overlap to the patch size, or it can be a non-negative int
            that defines number of pixels for overlap. Defaults to 0.0.
        filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
            return True for a patch to keep. Defaults to no filtering.
        pad_mode: string define the mode for `torch.nn.functional.pad`. The acceptable values are
            `"constant"`, `"reflect"`, `"replicate"`, `"circular"` or `None`. Default to `"constant"`.
            If None, no padding will be applied, so it will drop the patches crossing the border of
            the image (either when the offset is negative or the image is non-divisible by the patch_size).
        pad_value: the value for `"constant"` padding. Defaults to 0.
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
        pad_mode: str | None = "constant",
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
        # initialize with zero
        pad_size = [0] * 2 * spatial_ndim
        if not self.pad_mode:
            return pad_size, False
        # set the starting pad size only if the offset is negative
        pad_size[1::2] = (-min(off, 0) for off in offset)
        # set the ending pad size only if it is not divisible by the patch size
        end_padding = []
        for sh, off, ps, ov in zip(spatial_shape, offset, patch_size, overlap):
            if ps == 0:
                pad_amount = 0
            else:
                if isinstance(ov, float):
                    pad_amount = (off - sh + ps) % round(ps - (ps * ov))
                else:
                    pad_amount = (off - sh + ps) % round(ps - ov)
            end_padding.append(pad_amount)

        pad_size[::2] = end_padding
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
        slices = (slice(None),) * 2 + tuple(slice(loc, loc + ps) for loc, ps in zip(location, patch_size))
        return inputs[slices]

    def get_input_shape(self, inputs: Any) -> tuple:
        """
        Return the input spatial shape.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Returns:
            spatial_shape
        """
        return tuple(inputs.shape[2:])

    def get_padded_shape(self, inputs: Any) -> tuple:
        """
        Return the actual spatial shape covered by the output split patches.
        For instance, if the input image is padded, the actual spatial shape will be enlarged
        and not the same as input spatial shape.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Returns:
            padded_spatial_shape

        """
        spatial_shape = self.get_input_shape(inputs)
        if not self.pad_mode:
            return spatial_shape
        spatial_ndim = len(spatial_shape)
        patch_size, overlap, offset = self._get_valid_shape_parameters(spatial_shape)
        pad_size, _ = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)
        padded_spatial_shape = tuple(ss + ps + pe for ss, ps, pe in zip(spatial_shape, pad_size[1::2], pad_size[::2]))

        return padded_spatial_shape

    def __call__(self, inputs: Any) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: either a torch.Tensor with BCHW[D] dimensions, representing an image or a batch of images

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """

        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"The input should be a tensor. {type(inputs)} is given.")

        spatial_shape = inputs.shape[2:]
        spatial_ndim = len(spatial_shape)
        patch_size, overlap, offset = self._get_valid_shape_parameters(spatial_shape)
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        # Padding
        if self.pad_mode and any(pad_size):
            # pad the inputs
            inputs = torch.nn.functional.pad(inputs, pad_size[::-1], mode=self.pad_mode, value=self.pad_value)
            # update spatial shape
            spatial_shape = inputs.shape[2:]
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        # Splitting
        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            patch = self._get_patch(inputs, location, patch_size)
            patch = convert_to_tensor(patch, device=self.device)
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patch and yield
            if self.filter_fn is None or self.filter_fn(patch, location):
                yield patch, location


class WSISlidingWindowSplitter(SlidingWindowSplitter):
    """
    Splits the whole slide image input into patches with sliding window strategy and a possible overlap.
    This extracts patches from file without loading the entire slide into memory.
    It also allows offsetting the starting position and filtering the patches.

    Args:
        patch_size : the size of the patches to be generated.
        offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
        overlap: the amount of overlap between patches in each dimension. It can be either a float in
            the range of [0.0, 1.0) that defines relative overlap to the patch size, or it can be a non-negative int
            that defines number of pixels for overlap. Defaults to 0.0.
        filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
            return True for a patch to keep. Defaults to no filtering.
        pad_mode: define the mode for padding. Either "constant" or None. Default to "constant".
            Padding is only supported with "OpenSlide" or "cuCIM" backend, and the filling value is 256.
        device: the device where the patches are generated. Defaults to the device of inputs.
        reader: the module to be used for loading whole slide imaging. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`. Defaults to "OpenSlide".
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
            - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

            To obtain an optimized performance please use either "cuCIM" or "OpenSlide" backend.
        reader_kwargs: the arguments to pass to `WSIReader` or the provided whole slide reader class.
            For instance, level=2, dtype=torch.float32, etc.
            Note that if `level` is not provided, `level=0` is assumed.

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
        pad_mode: str | None = "constant",
        device: torch.device | str | None = None,
        reader: str | BaseWSIReader | type[BaseWSIReader] | None = "OpenSlide",
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
        if self.reader.backend.lower() not in ["openslide", "cucim"]:
            warnings.warn(
                f"WSIReader with {self.reader.backend.lower()} backend is not supported for efficiently loading patches. "
                "This may cause an significant slow down and a large memory foot print. "
                "Please use other backends such as 'OpenSlide' or 'cuCIM' instead."
            )

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

    def get_input_shape(self, inputs: Any) -> tuple:
        """
        Return the input spatial shape.

        Args:
            inputs: either a tensor of shape BCHW[D], representing a batch of images,
                or a filename (str) or list of filenames to the image(s).

        Returns:
            spatial_shape

        """
        wsi = self.reader.read(inputs)
        level = self.reader_kwargs.get("level", 0)
        return self.reader.get_size(wsi, level)

    def __call__(self, inputs: PathLike | Sequence[PathLike]) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: the file path to a whole slide image.

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """
        # Handle if the input file paths are batched
        if not isinstance(inputs, str) and isinstance(inputs, Sequence):
            if len(inputs) > 1:
                raise ValueError("Only batch size of one would work for wsi image. Please provide one path at a time.")
            inputs = inputs[0]

        # Check if the input is a sting or path like
        if not isinstance(inputs, (str, os.PathLike)):
            raise ValueError(f"The input should be the path to the whole slide image. {type(inputs)} is given.")

        wsi = self.reader.read(inputs)
        level = self.reader_kwargs.get("level", 0)
        downsample_ratio = self.reader.get_downsample_ratio(wsi, level)
        spatial_shape: tuple = self.reader.get_size(wsi, level)
        spatial_ndim = len(spatial_shape)
        if spatial_ndim != 2:
            raise ValueError(f"WSIReader only support 2D images. {spatial_ndim} spatial dimension is provided.")
        patch_size, overlap, offset = self._get_valid_shape_parameters(spatial_shape)
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        # Padding (extend the spatial shape)
        if any(pad_size):
            spatial_shape = tuple(ss + ps + pe for ss, ps, pe in zip(spatial_shape, pad_size[1::2], pad_size[::2]))
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        # Splitting (extracting patches)
        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            location_ = tuple(round(loc * downsample_ratio) for loc in location)
            patch = self._get_patch(wsi, location_, patch_size)
            patch = convert_to_tensor(patch, device=self.device)
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patch and yield
            if self.filter_fn is None or self.filter_fn(patch, location):
                yield patch, location
