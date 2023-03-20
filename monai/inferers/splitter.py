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
from inspect import _empty, isclass, signature
from typing import Any

import torch

from monai.config import PathLike
from monai.data.utils import iter_patch_position
from monai.data.wsi_reader import BaseWSIReader, WSIReader
from monai.transforms import ToTensor
from monai.utils.misc import ensure_tuple, ensure_tuple_rep

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
    """Split the input into patches with sliding window strategy and a possible overlap.
    It also allows to offset the starting position and filter the patches.

    Args:
        patch_size : the size of the patches to be generated.
        patch_level: [Only for WSI] the level at which the patches are extracted. Default to 0.
        offset: the amount of offset for the patches with respect to the original input.  Defaults to 0.
        overlap: the amount of overlap between patches in each dimension [0, 1). Defaults to 0.0.
        filter_fn: a callable to filter patches. It should accepts exactly two parameters (patch, location), and
            return True for a patch to keep. Defaults to no filtering.
        device: the device where the patches are generated. Defaults to the device of inputs.
        non_spatial_ndim: number of non-spatial dimensions (e.g. batch, color)
        pad_kwargs: arguments for `torch.nn.functional.pad`.
            To pad the input images in order to capture the patches crossing the border of the image
            (either when the offset is negative or the image is non-divisible by the patch_size),
            "mode" need to be set, for instance {"mode": "constant"}.
        reader: the module to be used for loading whole slide imaging. If `reader` is

            - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
            - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
            - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.

        reader_kwargs: additional arguments to pass to `WSIReader` or provided whole slide reader class

    Note:
        If only one scaler value is provided for `patch_size`, `offset`, or `overlap`,
            it will be broadcasted to all the spatial dimensions.
    """

    def __init__(
        self,
        patch_size: Sequence[int] | int,
        patch_level: int = 0,
        offset: Sequence[int] | int = 0,
        overlap: Sequence[float] | float = 0.0,
        filter_fn: Callable | None = None,
        device: torch.device | str | None = None,
        non_spatial_ndim: int = 2,
        pad_kwargs: dict | None = None,
        reader: str | BaseWSIReader | type[BaseWSIReader] | None = None,
        reader_kwargs: dict | None = None,
    ) -> None:
        super().__init__(patch_size=patch_size, device=device)
        self.offset = offset
        if any(ov < 0 or ov >= 1 for ov in ensure_tuple(overlap)):
            raise ValueError(f"Overlap must be between 0 and 1 but {overlap} is given.")
        self.overlap = overlap
        self.filter_fn = self._validate_patch_filter_fn(filter_fn)
        self.non_spatial_ndim = non_spatial_ndim
        self.pad_kwargs = pad_kwargs if pad_kwargs else {}
        if "mode" not in self.pad_kwargs:
            self.pad_kwargs["mode"] = None
        self.patch_level = patch_level
        self._set_reader(reader, reader_kwargs)

    @staticmethod
    def _validate_patch_filter_fn(filter_fn):
        if callable(filter_fn):
            sig = signature(filter_fn)
            n_params = len(sig.parameters)
            num_pos_params = len([v for v in sig.parameters.values() if v.default is _empty])
            if n_params < 2:
                raise ValueError(
                    f"`patch_filter_fn` requires to accept at least two parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {n_params} parameters."
                )
            elif num_pos_params > 2:
                raise ValueError(
                    f"`patch_filter_fn` can have at most two positional parameters (patch, location)."
                    f"The provided callable ({filter_fn}) has {num_pos_params} positional parameters."
                )
        elif filter_fn is not None:
            raise ValueError(
                "`patch_filter_fn` should be a callable with two input parameters (patch, location). "
                f"{type(filter_fn)} is given."
            )
        return filter_fn

    def _calculate_pad_size(self, spatial_shape, spatial_ndim, patch_size, offset, overlap):
        if not self.pad_kwargs["mode"]:
            return [], False
        # initialize with zero
        pad_size = [0] * 2 * spatial_ndim
        # set the starting pad size only if the offset is negative
        pad_size[1::2] = (-min(off, 0) for off in offset)
        # set the ending pad size only if it is not divisible by the patch size
        pad_size[::2] = (
            0 if ps == 0 else (off - sh + ps) % round(ps * (1.0 - ov))
            for sh, off, ps, ov in zip(spatial_shape, offset, patch_size, overlap)
        )
        return pad_size, any(pad_size[1::2])

    @staticmethod
    def _get_valid_sliding_window_params(patch_size, overlap, offset, spatial_shape):
        spatial_ndim = len(spatial_shape)
        # patch_size
        patch_size = ensure_tuple_rep(patch_size, spatial_ndim)
        # overlap
        overlap = ensure_tuple_rep(overlap, spatial_ndim)
        overlap = tuple(o if p else 0.0 for o, p in zip(overlap, patch_size))
        # offset
        offset = ensure_tuple_rep(offset, spatial_ndim)
        for off, ps, sh in zip(offset, patch_size, spatial_shape):
            if off < -ps:
                raise ValueError(f"Negative `offset` ({off}) cannot be larger than `patch_size` ({ps}) in magnitude.")
            if off >= sh:
                raise ValueError(f"`offset` ({off}) cannot be larger than inputs size ({sh}).")

        return patch_size, overlap, offset

    def _get_patch_tensor(self, inputs, location, patch_size):
        slices = (slice(None),) * self.non_spatial_ndim + tuple(
            slice(loc, loc + ps) for loc, ps in zip(location, patch_size)
        )
        patch = inputs[slices]
        # send the patch to target device
        patch.to(self.device)
        return patch

    def _get_patch_wsi(self, inputs, location, patch_size):
        patch, _ = self.reader.get_data(wsi=inputs, location=location, size=patch_size, level=self.patch_level)
        # send the patch to target device
        return ToTensor(device=self.device)(patch)

    def _set_reader(self, reader: str | BaseWSIReader | type[BaseWSIReader], reader_kwargs: dict | None) -> None:
        """
        Set the WSI reader object based on the input reader

        Args:
            reader: the module to be used for loading whole slide imaging. If `reader` is

                - a string, it defines the backend of `monai.data.WSIReader`. Defaults to cuCIM.
                - a class (inherited from `BaseWSIReader`), it is initialized and set as wsi_reader.
                - an instance of a class inherited from `BaseWSIReader`, it is set as the wsi_reader.
        """
        self.reader: WSIReader | BaseWSIReader | None = None
        self.reader_kwargs = {} if reader_kwargs is None else reader_kwargs
        if isinstance(reader, str):
            self.reader = WSIReader(backend=reader, level=self.patch_level, **self.reader_kwargs)
        elif isclass(reader) and issubclass(reader, BaseWSIReader):
            self.reader = reader(level=self.patch_level, **self.reader_kwargs)
        elif isinstance(reader, BaseWSIReader):
            self.reader = reader
        elif reader is not None:
            raise ValueError(f"Unsupported reader type: {reader}.")

    def __call__(self, inputs: torch.Tensor | PathLike) -> Iterable[tuple[torch.Tensor, Sequence[int]]]:
        """Split the input tensor into patches and return patches and locations.

        Args:
            inputs: a torch.Tensor with BCHW[D] dimensions, representing an image or a batch of images.

        Yields:
            tuple[torch.Tensor, Sequence[int]]: yields tuple of patch and location
        """
        if isinstance(inputs, torch.Tensor):
            self.get_patch = self._get_patch_tensor
            spatial_shape = inputs.shape[self.non_spatial_ndim :]
        elif isinstance(inputs, PathLike):
            self.get_patch = self._get_patch_wsi
            inputs = self.reader.read(inputs)
            if not isinstance(inputs, list):
                inputs = [inputs]
            spatial_shape = self.reader.get_size(inputs[0])
            for wsi in inputs[1:]:
                if self.reader.get_size(wsi) != spatial_shape:
                    raise RuntimeError(
                        "Spatial shape of the batched inputs should be the same."
                        f" {self.reader.get_size(wsi)} != {spatial_shape}."
                        "Please either provide one filename for whole slide image at a time "
                        "or make sure that the list of whole slides has the same spatial size."
                    )
        spatial_ndim = len(spatial_shape)
        patch_size, overlap, offset = self._get_valid_sliding_window_params(
            patch_size=self.patch_size, overlap=self.overlap, offset=self.offset, spatial_shape=spatial_shape
        )
        # check if there is any negative offset, there has to be a valid padding mode
        if any(off < 0 for off in offset) and not self.pad_kwargs["mode"]:
            raise ValueError(
                f"Negative `offset`requires a valid padding mode but `mode` is set to {self.pad_kwargs['mode']}."
            )
        pad_size, is_start_padded = self._calculate_pad_size(spatial_shape, spatial_ndim, patch_size, offset, overlap)

        if any(pad_size):
            if isinstance(inputs, torch.Tensor):
                # pad the inputs
                inputs = torch.nn.functional.pad(inputs, pad_size[::-1], **self.pad_kwargs)
                # update spatial shape
                spatial_shape = inputs.shape[self.non_spatial_ndim :]
            else:
                spatial_shape = [ss + ps for ss, ps in zip(spatial_shape, pad_size)]
            # correct the offset with respect to the padded image
            if is_start_padded:
                offset = tuple(off + p for off, p in zip(offset, pad_size[1::2]))

        for location in iter_patch_position(spatial_shape, patch_size, offset, overlap, False):
            patch = self.get_patch(inputs, location, patch_size)
            # correct the location with respect to original inputs (remove starting pads)
            if is_start_padded:
                location = tuple(loc - p for loc, p in zip(location, pad_size[1::2]))
            # filter patch and yield
            if self.filter_fn is None:
                yield patch, location
            elif self.filter_fn(patch, location):
                yield patch, location
