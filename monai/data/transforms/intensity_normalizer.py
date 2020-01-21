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

import numpy as np

from .transform import Transform


class IntensityNormalizer(Transform):
    """Normalize input based on provided args, using calculated mean and std if not provided
    (shape of subtrahend and divisor must match. if 0, entire volume uses same subtrahend and
     divisor, otherwise the shape can have dimension 1 for channels).
     Current implementation can only support 'channel_last' format data.

    Args:
        apply_keys (a hashable key or a tuple/list of hashable keys): run transform on which field of the input data
        subtrahend (ndarray): the amount to subtract by (usually the mean)
        divisor (ndarray): the amount to divide by (usually the standard deviation)
        dtype: output data format
    """

    def __init__(self, apply_keys, subtrahend=None, divisor=None, dtype=np.float32):
        _apply_keys = apply_keys if isinstance(apply_keys, (list, tuple)) else (apply_keys,)
        if not _apply_keys:
            raise ValueError('must set apply_keys for this transform.')
        for key in _apply_keys:
            if not isinstance(key, Hashable):
                raise ValueError('apply_keys should be a hashable or a sequence of hashables used by data[key]')
        self.apply_keys = _apply_keys
        if subtrahend is not None or divisor is not None:
            assert isinstance(subtrahend, np.ndarray) and isinstance(divisor, np.ndarray), \
                'subtrahend and divisor must be set in pair and in numpy array.'
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.dtype = dtype

    def __call__(self, data):
        assert data is not None and isinstance(data, dict), 'data must be in dict format with keys.'
        for key in self.apply_keys:
            img = data[key]
            assert key in data, 'can not find expected key={} in data.'.format(key)
            if self.subtrahend is not None and self.divisor is not None:
                img -= self.subtrahend
                img /= self.divisor
            else:
                img -= np.mean(img)
                img /= np.std(img)

            if self.dtype != img.dtype:
                img = img.astype(self.dtype)
            data[key] = img
        return data
