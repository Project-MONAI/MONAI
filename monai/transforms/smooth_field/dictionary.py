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

from monai.config import KeysCollection
from monai.transforms.smooth_field.array import RandSmoothFieldAdjustContrast, RandSmoothFieldAdjustIntensity
from monai.transforms.transform import MapTransform, RandomizableTransform, Transform
from monai.utils import InterpolateMode
from monai.utils.enums import TransformBackends

__all__ = ["RandSmoothFieldAdjustContrastd", "RandSmoothFieldAdjustIntensityd"]


class RandSmoothFieldAdjustContrastd(RandomizableTransform, MapTransform):
    """
    Dictionary version of RandSmoothFieldAdjustContrast. The field is randomized once per invocation by default so the
    same field is applied to every selected key.

    Args:
        keys: key names to apply the augment to
        spatial_size: size of input arrays, all arrays stated in `keys` must have same dimensions
        rand_size: size of the randomized field to start from
        padder: optional transform to add padding to the randomized field
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range for exponential field
        apply_same_field: if True, apply the same field to each key, otherwise randomize individually
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        rand_size: Union[Sequence[int], int],
        padder: Optional[Transform] = None,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.5, 4.5),
        apply_same_field: bool = True,
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.trans = RandSmoothFieldAdjustContrast(
            spatial_size=spatial_size,
            rand_size=rand_size,
            padder=padder,
            mode=mode,
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
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
        padder: optional transform to add padding to the randomized field
        mode: interpolation mode to use when upsampling
        align_corners: if True align the corners when upsampling field
        prob: probability transform is applied
        gamma: (min, max) range of intensity multipliers
        apply_same_field: if True, apply the same field to each key, otherwise randomize individually
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        rand_size: Union[Sequence[int], int],
        padder: Optional[Transform] = None,
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        prob: float = 0.1,
        gamma: Union[Sequence[float], float] = (0.1, 1.0),
        apply_same_field: bool = True,
    ):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)

        self.trans = RandSmoothFieldAdjustIntensity(
            spatial_size=spatial_size,
            rand_size=rand_size,
            padder=padder,
            mode=mode,
            align_corners=align_corners,
            prob=1.0,
            gamma=gamma,
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
