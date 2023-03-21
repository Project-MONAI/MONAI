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
A collection of dictionary-based wrappers around the "vanilla" transforms for crop and pad operations
defined in :py:class:`monai.transforms.croppad.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from monai.transforms.spatial.functional import get_pending_shape
from monai.transforms.traits import RandomizableTrait
from monai.transforms.croppad.functional import croppad
from monai.transforms.croppad.randomizer import CropRandomizer
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.lazy.functional import invert
from monai.transforms.transform import MapTransform, LazyTransform
from monai.utils import ensure_tuple_rep, GridSamplePadMode

__all__ = [
    # "Padd",
    # "SpatialPadd",
    # "BorderPadd",
    # "DivisiblePadd",
    # "Cropd",
    # "RandCropd",
    # "SpatialCropd",
    # "CenterSpatialCropd",
    # "CenterScaleCropd",
    # "RandScaleCropd",
    # "RandSpatialCropd",
    # "RandSpatialCropSamplesd",
    # "CropForegroundd",
    # "RandWeightedCropd",
    # "RandCropByPosNegLabeld",
    # "ResizeWithPadOrCropd",
    # "BoundingRectd",
    # "RandCropByLabelClassesd",
    # "PadD",
    # "PadDict",
    # "SpatialPadD",
    # "SpatialPadDict",
    # "BorderPadD",
    # "BorderPadDict",
    # "DivisiblePadD",
    # "DivisiblePadDict",
    # "CropD",
    # "CropDict",
    # "RandCropD",
    # "RandCropDict",
    # "SpatialCropD",
    # "SpatialCropDict",
    # "CenterSpatialCropD",
    # "CenterSpatialCropDict",
    # "CenterScaleCropD",
    # "CenterScaleCropDict",
    # "RandScaleCropD",
    # "RandScaleCropDict",
    # "RandSpatialCropD",
    # "RandSpatialCropDict",
    # "RandSpatialCropSamplesD",
    # "RandSpatialCropSamplesDict",
    # "CropForegroundD",
    # "CropForegroundDict",
    # "RandWeightedCropD",
    # "RandWeightedCropDict",
    # "RandCropByPosNegLabelD",
    # "RandCropByPosNegLabelDict",
    # "ResizeWithPadOrCropD",
    # "ResizeWithPadOrCropDict",
    # "BoundingRectD",
    # "BoundingRectDict",
    # "RandCropByLabelClassesD",
    # "RandCropByLabelClassesDict",
    "CropPadd",
    "RandCropPadd",
]


class CropPadd(MapTransform, InvertibleTransform, LazyTransform):

    def __init__(
            self,
            keys,
            slices=None,
            starts=None,
            ends=None,
            padding_mode=GridSamplePadMode.BORDER,
            allow_missing_keys=False,
            lazy_evaluation: bool = True
            ):
        MapTransform.__init__(self, keys, allow_missing_keys=allow_missing_keys)
        LazyTransform.__init__(self, lazy_evaluation)
        self.slices = slices
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def __call__(self, data):

        rd = dict(data)

        for key, _padding_mode in self.key_iterator(rd, self.padding_mode):
            rd[key] = croppad(rd[key], slices=self.slices, padding_mode=_padding_mode,
                              lazy_evaluation=self.lazy_evaluation)

        return rd

    def inverse(self, data):
        rd = dict(data)

        for key in self.key_iterator(rd):
            rd[key] = invert(rd[key], self.lazy_evaluation)
        return rd


class RandCropPadd(MapTransform, InvertibleTransform, LazyTransform, RandomizableTrait):

    def __init__(
            self,
            keys,
            sizes: Sequence[int] | int,
            padding_mode: GridSamplePadMode | str = GridSamplePadMode.BORDER,
            allow_missing_keys=False,
            lazy_evaluation: bool = True,
            seed = None,
            state = None
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        LazyTransform.__init__(self, lazy_evaluation)
        # self.sizes = sizes
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

        self.randomizer = CropRandomizer(sizes, seed=seed, state=state)

    def __call__(
            self,
            data: torch.Tensor,
    ):
        rd = dict(data)

        # TODO: this should be parameterized so that it can crop images with differing extents
        extents = None

        for key_, padding_mode_, in self.key_iterator(rd, self.padding_mode):
            if extents is None:
                img_shape = get_pending_shape(data[key_])[1:]
                print("lazy pre", self.randomizer.R.rand())
                extents = self.randomizer.sample(img_shape)
                print("lazy post", self.randomizer.R.rand())

            rd[key_] = croppad(data[key_], extents, padding_mode_, lazy_evaluation=self.lazy_evaluation)

        return rd

    def set_random_state(
            self,
            seed: int | None = None,
            state: np.random.RandomState | None = None
    ):
        self.randomizer.set_random_state(seed, state)
        return self

    def inverse(self, data):
        return invert(data, self.lazy_evaluation)


CropPadD = CropPadDict = CropPadd
RandCropPadD = RandCropPadDict = RandCropPadd
# PadD = PadDict = Padd
# SpatialPadD = SpatialPadDict = SpatialPadd
# BorderPadD = BorderPadDict = BorderPadd
# DivisiblePadD = DivisiblePadDict = DivisiblePadd
# CropD = CropDict = Cropd
# RandCropD = RandCropDict = RandCropd
# SpatialCropD = SpatialCropDict = SpatialCropd
# CenterSpatialCropD = CenterSpatialCropDict = CenterSpatialCropd
# CenterScaleCropD = CenterScaleCropDict = CenterScaleCropd
# RandSpatialCropD = RandSpatialCropDict = RandSpatialCropd
# RandScaleCropD = RandScaleCropDict = RandScaleCropd
# RandSpatialCropSamplesD = RandSpatialCropSamplesDict = RandSpatialCropSamplesd
# CropForegroundD = CropForegroundDict = CropForegroundd
# RandWeightedCropD = RandWeightedCropDict = RandWeightedCropd
# RandCropByPosNegLabelD = RandCropByPosNegLabelDict = RandCropByPosNegLabeld
# RandCropByLabelClassesD = RandCropByLabelClassesDict = RandCropByLabelClassesd
# ResizeWithPadOrCropD = ResizeWithPadOrCropDict = ResizeWithPadOrCropd
# BoundingRectD = BoundingRectDict = BoundingRectd
