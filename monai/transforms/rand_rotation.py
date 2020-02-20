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

import copy

import numpy as np

import monai

export = monai.utils.export("monai.transforms")


class Randomizable:
    """
    provide a `randomize` method for transforms, so that the transforms
    can update their internal states with some random factors. this is
    mainly for randomized data augmentation transforms.
    """

    def randomize(self):
        raise NotImplementedError


@export
class Rotate90:
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k=1, axes=(1, 2)):
        """
        Args:
            k (int): number of times to rotate by 90 degrees.
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        self.k = k
        self.plane_axes = axes

    def __call__(self, img):
        return np.rot90(img, self.k, self.plane_axes)


@export
class RandRotate90(Randomizable):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.axes = axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self):
        self._do_transform = np.random.random() < self.prob
        self._rand_k = np.random.randint(self.max_k) + 1
        return self

    def __call__(self, img):
        if not self._do_transform:
            return img
        return Rotate90(self._rand_k, self.axes)(img)


@export
def fn_map(transform, map_key=None, common_key=None, inplace=True):
    """
    Convert a vanilla `transform` into a dictionary-based transform.

    The new transform takes a dictionary as input and returns a dictionary.
    In the returned dictionay, data[map_key] will be transformed by `transform`,
    with data[common_key] as additional arguments.

    Args:
        map_key (str or list/tuple of str): keys to apply `transform`.
        common_key (str or list/tuple of str): keys of additional arguments to be passed to `tranform` from data dict.
        inplace (bool): whether to modify the data dict inplace.
    """
    if map_key:
        _map_key = map_key if isinstance(map_key, (list, tuple)) else (map_key,)
    else:
        _map_key = None
    if common_key:
        _common_key = common_key if isinstance(common_key, (list, tuple)) else (common_key,)
    else:
        _common_key = None

    def _dict_transform(data):
        if not isinstance(data, dict):
            raise ValueError('{} is not a dictionary', data)
        d = copy.copy(data) if not inplace else data

        common_args = {name: d.get(name, None) for name in _common_key} if _common_key else None
        apply_key = _map_key or tuple(d)
        for key in apply_key:
            if common_args:
                d[key] = transform(d[key], **common_args)
            else:
                d[key] = transform(d[key])
        return d

    return _dict_transform


if __name__ == "__main__":
    data = {
        'img': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
        'seg': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
        'affine': 3,
        'dtype': 4,
        'unused': 5,
    }
    rotator = RandRotate90(0.8).randomize()
    data_result = fn_map(rotator, map_key=['img', 'seg'])(data)
    print(data_result.keys())
    print(data_result['img'], data_result['seg'])
