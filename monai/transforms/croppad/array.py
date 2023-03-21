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
"""
A collection of "vanilla" transforms for crop and pad operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.traits import RandomizableTrait, MultiSampleTrait
from monai.transforms.croppad.functional import (
    pad,
    croppad
)
from monai.transforms.croppad.randomizer import CropRandomizer
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import invert
from monai.transforms.transform import LazyTransform
from monai.utils import GridSamplePadMode


__all__ = [
    "CropPad",
    "RandomCropPad",
    "RandomCropPadMultiSample",
]


class Pad(InvertibleTransform, LazyTransform):

    def __init__(
            self,
            padding: Sequence[Sequence[int, int]] | Sequence[int, int],
            mode: str | None = "border",
            value: int | float = 0,
            lazy_evaluation: bool = True
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.padding = padding
        self.padding_mode = mode
        self.value = value

    def __call__(
            self,
            img: torch.Tensor,
            padding: Sequence[tuple] | tuple | None = None,
            mode: str | None = None,
            value: int | float | None = None
    ):
        padding_ = self.padding if padding is None else padding
        padding_mode_ = self.padding_mode if mode is None else mode
        value_ = self.value if value is None else value

        img_t = pad(img, padding_, padding_mode_, value_,
                    lazy_evaluation=self.lazy_evaluation)

        return img_t

    def __invert__(self, data):
        return invert(data, self.lazy_evaluation)


class CropPad(InvertibleTransform, LazyTransform):

    def __init__(
            self,
            slices: Sequence[slice] | None = None,
            starts: Sequence[float] | None = None,
            ends: Sequence[float] | None = None,
            padding_mode: GridSamplePadMode | str = GridSamplePadMode.BORDER,
            lazy_evaluation: bool = True
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        self.slices = slices
        if slices is None:
            self.starts = starts
            self.ends = ends
        self.padding_mode = padding_mode

    def __call__(
            self,
            img: NdarrayOrTensor,
            slices: Sequence[slice] | None = None,
            shape_override: Sequence | None = None
    ):
        slices_ = slices if self.slices is None else self.slices

        img_t = croppad(img, slices_, self.padding_mode, lazy_evaluation=self.lazy_evaluation)

        return img_t

    def inverse(self, data):
        return invert(data, self.lazy_evaluation)


class RandomCropPad(InvertibleTransform, LazyTransform, RandomizableTrait):

    def __init__(
            self,
            sizes: Sequence[int] | int,
            prob: float = 0.1,
            padding_mode: GridSamplePadMode | str = GridSamplePadMode.BORDER,
            lazy_evaluation: bool = True
    ):
        LazyTransform.__init__(self, lazy_evaluation)
        # self.sizes = sizes
        self.padding_mode = padding_mode

        self.randomizer = CropRandomizer(sizes)

    def __call__(
            self,
            img: torch.Tensor,
            randomize: bool = True
    ):

        img_shape = img.shape[1:]

        extents = self.randomizer.sample(img_shape)

        img_t = croppad(img, extents, self.padding_mode, lazy_evaluation=self.lazy_evaluation)

        return img_t

    def inverse(self, data):
        return invert(data, self.lazy_evaluation)


class RandomCropPadMultiSample(
    InvertibleTransform, LazyTransform, RandomizableTrait, MultiSampleTrait
):

    def __init__(
            self,
            sizes: Sequence[int] | int,
            sample_count: int,
            padding_mode: GridSamplePadMode | str = GridSamplePadMode.BORDER,
            lazy_evaluation: bool = True
    ):
        self.sample_count = sample_count
        self.op = RandomCropPad(sizes, 1.0, padding_mode, lazy_evaluation)

    def __call__(
            self,
            img: torch.Tensor,
            randomize: bool = True
    ):
        for i in range(self.sample_count):
            yield self.op(img, randomize)

    def inverse(
            self,
            data: NdarrayOrTensor
    ):
        raise NotImplementedError()

    def set_random_state(self, seed=None, state=None):
        self.op.set_random_state(seed, state)
