# Copyright 2020 MONAI Consortium
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
A collection of dictionary-based wrappers around the "vanilla" transforms
defined in `monai.transforms.transforms`.
"""

from collections.abc import Hashable

import monai
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.transforms import Rotate90
from monai.utils.misc import ensure_tuple

export = monai.utils.export("monai.transforms")


class MapTransform(Transform):
    """
    A subclass of ``monai.transforms.compose.Transform`` with an assumption
    that the ``data`` input of ``self.__call__`` is a MutableMapping such as ``dict``.

    The ``keys`` parameter will be used to get and set the actual data
    item to transform.  That is, the callable of this transform should
    follow the pattern:
    ```
        def __call__(self, data):
            for key in self.keys:
                if key in data:
                    update output data with some_transform_function(data[key]).
                else:
                    do nothing or some exceptions handling.
            return data
    ```
    """

    def __init__(self, keys):
        self.keys = ensure_tuple(keys)
        if not self.keys:
            raise ValueError('keys unspecified')
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise ValueError('keys should be a hashable or a sequence of hashables, got {}'.format(type(key)))


@export
class Rotate90d(MapTransform):
    """
    dictionary-based wrapper of Rotate90.
    """

    def __init__(self, keys, k=1, axes=(1, 2)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        MapTransform.__init__(self, keys)
        self.k = k
        self.plane_axes = axes

        self.rotator = Rotate90(self.k, self.plane_axes)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


@export
class RandRotate90d(Randomizable, MapTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, keys, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                See also: monai.transform.composables.MapTransform
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
                (Default to (1, 2))
        """
        MapTransform.__init__(self, keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.axes = axes

        self._do_transform = False
        self._rand_k = 0

    def randomise(self):
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomise()
        if not self._do_transform:
            return data

        rotator = Rotate90(self._rand_k, self.axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


# if __name__ == "__main__":
#     import numpy as np
#     data = {
#         'img': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
#         'seg': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
#         'affine': 3,
#         'dtype': 4,
#         'unused': 5,
#     }
#     rotator = RandRotate90d(keys=['img', 'seg'], prob=0.8)
#     # rotator.set_random_state(1234)
#     data_result = rotator(data)
#     print(data_result.keys())
#     print(data_result['img'], data_result['seg'])
