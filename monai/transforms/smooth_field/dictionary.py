# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from monai.config import KeysCollection
from monai.transforms.smooth_field.array import (
    RandSmoothDeform,
    RandSmoothFieldAdjustContrast,
    RandSmoothFieldAdjustIntensity,
)
from monai.transforms.transform import MapTransform, RandomizableTransform, Transform
from monai.utils import GridSampleMode, InterpolateMode
from monai.utils.enums import TransformBackends

__all__ = ["RandSmoothFieldAdjustContrastd", "RandSmoothFieldAdjustIntensityd", "RandSmoothDeformd"]


class RandSmoothFieldAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothFieldAdjustContrast. The field is randomized once per invocation by default so the
    same field is applied to every selected key.

    Args:
        keys: key names to apply the augment to
        spatial_size: size of input arrays, all arrays stated in `keys` must have same dimensions
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range for exponential field
        apply_same_field: if True, apply the same field to each key, otherwise randomize individually
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        rand_size: Union[Sequence[int], int],
        pad: int = 0,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.5, 4.5),
        apply_same_field: bool = True,
        device: Optional[torch.device] = None
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.trans = RandSmoothFieldAdjustContrast(
            spatial_size=spatial_size,
            rand_size=rand_size,
            pad=pad,
            mode=mode,
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
            device=device
        )
        self.apply_same_field = apply_same_field

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustContrastd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)

        if self._do_transform:
            self.trans.randomize()

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for key in self.key_iterator(d):
            if not self.apply_same_field:
                self.randomize()  # new field for every key

            d[key] = self.trans(d[key], False)

        return d


class RandSmoothFieldAdjustIntensityd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothFieldAdjustIntensity. The field is randomized once per invocation by default so
    the same field is applied to every selected key.

    Args:
        keys: key names to apply the augment to
        spatial_size: size of input arrays, all arrays stated in `keys` must have same dimensions
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range of intensity multipliers
        apply_same_field: if True, apply the same field to each key, otherwise randomize individually
        device: Pytorch device to define field on
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        rand_size: Union[Sequence[int], int],
        pad: int = 0,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.1, 1.0),
        apply_same_field: bool = True,
        device: Optional[torch.device] = None
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.trans = RandSmoothFieldAdjustIntensity(
            spatial_size=spatial_size,
            rand_size=rand_size,
            pad=pad,
            mode=mode,
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
            device=device
        )
        self.apply_same_field = apply_same_field

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustIntensityd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.trans.randomize()

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for key in self.key_iterator(d):
            if not self.apply_same_field:
                self.randomize()  # new field for every key

            d[key] = self.trans(d[key], False)

        return d

    
class RandSmoothDeformd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothDeform.
    
    Args:
        keys: key names to apply the augment to
        spatial_size: input array size to which deformation grid is interpolated
        rand_size: size of the randomized field to start from
        pad: number of pixels/voxels along the edges of the field to pad with 0
        field_mode: interpolation mode to use when upsampling the deformation field
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        def_range: (min, max) value of the deformation range in pixel/voxel units
        grid_dtype: type for the deformation grid calculated from the field
        grid_mode: interpolation mode used for sampling input using deformation grid
        grid_align_corners: if True align the corners when sampling the deformation grid
        apply_same_field: if True, apply the same field to each key, otherwise randomize individually
        device: Pytorch device to define field on
    """
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        rand_size: Union[Sequence[int], int],
        pad: int = 0,
        field_mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        def_range: float = 1.0,
        grid_dtype=torch.float32,
        grid_mode: Union[GridSampleMode, str] = GridSampleMode.NEAREST,
        grid_align_corners: Optional[bool] = False,
        apply_same_field: bool = True,
        device: Optional[torch.device] = None,        
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)
        
        self.trans=RandSmoothDeform(
            rand_size=rand_size,
            spatial_size=spatial_size,
            pad=pad,
            field_mode=field_mode,
            align_corners=align_corners,
            prob=1.0,
            def_range=def_range,
            grid_dtype=grid_dtype,
            grid_mode=grid_mode,
            grid_align_corners=grid_align_corners,
            device=device
        )
        
        self.apply_same_field=apply_same_field
        
    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSmoothFieldAdjustIntensityd":
        super().set_random_state(seed, state)
        self.trans.set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.trans.randomize()
        
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()

        if not self._do_transform:
            return data

        d = dict(data)

        for key in self.key_iterator(d):
            if not self.apply_same_field:
                self.randomize()  # new field for every key

            d[key] = self.trans(d[key], False)

        return d