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

import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.transforms import Spacing

TEST_CASES = [
    [{'pixdim': (1.0, 2.0, 1.5)},
     np.ones((2, 10, 15, 20)), {'original_pixdim': (0.5, 0.5, 1.0)}, (2, 5, 4, 13)],
    [{'pixdim': (1.0, 2.0, 1.5), 'keep_shape': True},
     np.ones((1, 2, 1, 2)), {'original_pixdim': (0.5, 0.5, 1.0)}, (1, 2, 1, 2)],
    [{'pixdim': (1.0, 0.2, 1.5), 'keep_shape': False},
     np.ones((1, 2, 1, 2)), {'original_affine': np.eye(4)}, (1, 2, 5, 1)],
    [{'pixdim': (1.0, 2.0), 'keep_shape': True},
     np.ones((3, 2, 2)), {'original_pixdim': (1.5, 0.5)}, (3, 2, 2)],
    [{'pixdim': (1.0, 0.2), 'keep_shape': False},
     np.ones((5, 2, 1)), {'original_pixdim': (1.5, 0.5)}, (5, 3, 2)],
    [{'pixdim': (1.0,), 'keep_shape': False},
     np.ones((1, 2)), {'original_pixdim': (1.5,), 'interp_order': 0}, (1, 3)],
]


class TestSpacingCase(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_spacing(self, init_param, img, data_param, expected_shape):
        res = Spacing(**init_param)(img, **data_param)
        np.testing.assert_allclose(res[0].shape, expected_shape)
        if 'original_pixdim' in data_param:
            np.testing.assert_allclose(res[1], data_param['original_pixdim'])
        np.testing.assert_allclose(res[2], init_param['pixdim'])


if __name__ == '__main__':
    unittest.main()
