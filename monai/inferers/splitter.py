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
from typing import Any

import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import iter_patch_position
from monai.utils.enums import PatchKeys, PytorchPadMode
from monai.utils.misc import ensure_tuple, ensure_tuple_rep
from monai.utils.module import look_up_option

__all__ = ["Splitter"]


class Splitter(ABC):
    """
    A base class for splitting the inputs into iterable patches (MetaTensor with PatchKeys metadata).
    Extend this class to support operations for `PatchInference`, e.g. SlidingPatchSplitter.

    """

    def __init__(self, patch_size: Sequence[int] | int, device: torch.device | str | None = None) -> None:
        self.patch_size = patch_size
        self.device = device

    @abstractmethod
    def __call__(self, inputs: Any) -> Iterable[MetaTensor]:
        """
        Split the image, represented by an input tensor or a filename, into patches.

        Args:
            inputs: either a tensor of shape BxCxHxW[xD], representing a batch of images,
                or a filename (str) or list of filenames to the image(s)

        Raises:
            NotImplementedError: When the subclass does not override this method.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class SlidingWindowSplitter(Splitter):
    def __init__(
        self,
        patch_size: Sequence[int] | int,
        offset: Sequence[int] | int = 0,
        device: torch.device | str | None = None,
        overlap: Sequence[float] | float = 0.0,
        patch_filter_fn: Callable | None = None,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        super().__init__(patch_size=patch_size, device=device)
        self.offset = offset
        self.pad_mode = pad_mode
        self.pad_kwargs = pad_kwargs
        if any(ov < 0 or ov >= 1 for ov in ensure_tuple(overlap)):
            raise ValueError("Overlap must be between 0 and 1.")
        self.overlap = overlap

    def __call__(self, inputs: torch.Tensor) -> Iterable[MetaTensor]:
        if self.device:
            inputs.to(self.device)
        spatial_ndim = inputs.ndim - 2
        patch_size = ensure_tuple_rep(self.patch_size, spatial_ndim)
        overlap = ensure_tuple_rep(self.overlap, spatial_ndim)
        overlap = tuple(o if p else 0.0 for o, p in zip(overlap, patch_size))  # overlap only in patching dimensions
        offset = ensure_tuple_rep(self.offset, spatial_ndim)
        for off, ps, ins in zip(offset, patch_size, inputs.shape[2:]):
            if off < -ps:
                raise ValueError(f"Negative `offset` ({off}) cannot be larger than `patch_size` ({ps}) in magnitude.")
            if off >= ins:
                raise ValueError(f"`offset` ({off}) cannot be larger than inputs size ({ins}).")

        padded = bool(self.pad_mode)
        _pad_size = [0] * 2 * spatial_ndim
        if padded:
            # set the starting pad size only if the offset is negative
            _pad_size[1::2] = (-min(off, 0) for off in offset)
            # set the ending pad size only if it is not divisible by the patch size
            _pad_size[::2] = (
                0 if ps == 0 else (off - ins + ps) % round(ps * (1.0 - ov))
                for ins, off, ps, ov in zip(inputs.shape[2:], offset, patch_size, overlap)
            )
            # pad the inputs
            inputs = torch.nn.functional.pad(
                inputs, _pad_size[::-1], look_up_option(self.pad_mode, PytorchPadMode).value, **self.pad_kwargs
            )
            # correct the offset with regard to the padded image
            offset = tuple(off + p for off, p in zip(offset, _pad_size[1::2]))

        for location in iter_patch_position(inputs.shape[2:], patch_size, offset, overlap, False):
            slices = (slice(None),) * 2 + tuple(slice(loc, loc + ps) for loc, ps in zip(location, patch_size))
            patch = inputs[slices]
            if padded:
                # correct the location for original inputs (remove padding)
                location = tuple(loc - p for loc, p in zip(location, _pad_size[1::2]))
            # metadata = patch.meta if isinstance(patch, MetaTensor) else MetaTensor.get_default_meta()
            # metadata[PatchKeys.LOCATION] = torch.tensor(location)
            # metadata[PatchKeys.SIZE] = torch.tensor(patch_size)
            # yield MetaTensor(x=patch, meta=metadata)
            yield patch, location
