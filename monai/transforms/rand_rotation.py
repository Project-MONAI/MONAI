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

from collections.abc import Hashable
from copy import copy

import numpy as np

import monai

export = monai.utils.export("monai.transforms")


class Randomizable:
    """
    An interface for handling local random state.
    this is mainly for randomized data augmentation transforms.
    """
    R = np.random.RandomState()

    def set_random_state(self, seed=None, state=None):
        """
        Set the random state locally, to control the randomness, the derived
        classes should use `self.R` instead of `np.random` to introduce random
        factors.

        Args:
            seed (int): set the random state with an integer seed.
            state (np.random.RandomState): set the random state with a `np.random.RandomState` object.

        Note:
            thread-safty
        """
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, int) else seed
            self.R = np.random.RandomState(_seed)
            return

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise ValueError('`state` must be a `np.random.RandomState`, got {}'.format(type(state)))
            self.R = state
            return

        self.R = np.random.RandomState()
        return

    def randomise(self):
        """
        all self.R calls happen here so that we have a better chance to identify errors of sync the random state.
        """
        raise NotImplementedError


class Transform(object):
    """
    An abstract class of a ``Transform``.
    A transform is callable that processes ``data``.

    It could be stateful and modify ``data`` in place.
    Therefore the implementation should be aware of:
    - thread-safety when mutating its own states.
        When used from a multi-process context, this class's states are read-only.
    - ``data`` content unused by this transform may still be used in the
        subsequent transforms in a composed transform.
        see also: `monai.transforms.compose.Compose`.
    - storing too much information in ``data`` may not scale.
    """

    def __call__(self, data):
        """
        ``data`` is an element which often comes from an iteration over an
        iterable, such as``torch.utils.data.Dataset``. This method should
        return an updated version of ``data``.
        """
        raise NotImplementedError


class MapTransform(Transform):
    """
    A subclass of ``Transform`` with an assumption that the ``data`` input of
    ``self.__call__`` is a MutableMapping such as ``dict``.

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
        self.keys = keys if isinstance(keys, (list, tuple)) else (keys,)
        if not self.keys:
            raise ValueError('keys unspecified')
        for key in self.keys:
            if not isinstance(key, Hashable):
                raise ValueError('keys should be a hashable or a list of hashables, got {}'.format(type(key)))


@export
class Rotate90(Transform):
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

    def __call__(self, data):
        d = copy(data)
        for key in self.keys:
            if key in d:
                d[key] = Rotate90(self.k, self.plane_axes)(d[key])
            else:
                raise KeyError('data dict does contain {} key'.format(key))
        return d


@export
class RandRotate90(Randomizable, MapTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `axes`.
    """

    def __init__(self, keys, prob=0.1, max_k=3, axes=(1, 2)):
        """
        Args:
            keys (hashable items): keys of the corresponding items to be transformed.
                see also: monai.transform.rand_rotation.MapTransform
            prob (float): probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k (int): number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            axes (2 ints): defines the plane to rotate with 2 axes.
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
        return Rotate90d(self.keys, self._rand_k, self.axes)(data)


if __name__ == "__main__":
    data = {
        'img': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
        'seg': np.array((1, 2, 3, 4)).reshape((1, 2, 2)),
        'affine': 3,
        'dtype': 4,
        'unused': 5,
    }
    rotator = RandRotate90(keys=['img', 'seg'], prob=0.8)
    # rotator.set_random_state(1234)
    data_result = rotator(data)
    print(data_result.keys())
    print(data_result['img'], data_result['seg'])
