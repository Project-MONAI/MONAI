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
    [
        {'pixdim': (2.0, )},
        np.ones((1, 2)),  # data
        {'original_affine': np.eye(4)},
        np.array([[1., 0.]])
    ],
    [
        {'pixdim': (1.0, 0.2, 1.5)},
        np.ones((1, 2, 1, 2)),  # data
        {'original_affine': np.eye(4)},
        np.array([[[[1., 0.]], [[1., 0.]]]])
    ],
    [
        {'pixdim': (1.0, 0.2, 1.5), 'diagonal': False},
        np.ones((1, 2, 1, 2)),  # data
        {
            'original_affine': np.array([[2, 1, 0, 4], [-1, -3, 0, 5], [0, 0, 2., 5], [0, 0, 0, 1]],),
        },
        np.array([[[[0., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]], [[0., 0., 0.]]]])
    ],
    [
        {'pixdim': (3.0, 1.0)},
        np.arange(24).reshape((2, 3, 4)),  # data
        {'original_affine': np.diag([-3.0, 0.2, 1.5, 1])},
        np.array([[[0, 0], [4, 0], [8, 0]], [[12, 0], [16, 0], [20, 0]]])
    ],
    [
        {'pixdim': (3.0, 1.0)},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array([[[0, 1, 2, 3], [0, 0, 0, 0]], [[12, 13, 14, 15], [0, 0, 0, 0]]])
    ],
    [
        {'pixdim': (1.0, 1.0)},
        np.arange(24).reshape((2, 3, 4)),  # data
        {},
        np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ],
    [
        {'pixdim': (4.0, 5.0, 6.0)},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {'original_affine': np.array([[-4, 0, 0, 4], [0, 5, 0, -5], [0, 0, 6, -6], [0, 0, 0, 1]])},
        np.array([[[[17, 18, 19, 0], [21, 22, 23, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]])
    ],
    [
        {'pixdim': (4.0, 5.0, 6.0)},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {'original_affine': np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array([[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]])
    ],
    [
        {'pixdim': (4.0, 5.0, 6.0), 'mode': 'nearest'},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {'original_affine': np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]])},
        np.array([[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                   [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]])
    ],
    [
        {'pixdim': (4.0, 5.0, 6.0), 'mode': 'nearest'},
        np.arange(24).reshape((1, 2, 3, 4)),  # data
        {'original_affine': np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), 'interp_order': 0},
        np.array([[[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                   [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]])
    ],
    [
        {'pixdim': (2.0, 5.0, 6.0), 'mode': 'constant'},
        np.arange(24).reshape((1, 4, 6)),  # data
        {'original_affine': np.array([[-4, 0, 0, 4], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), 'interp_order': 0},
        np.array([[[6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23],
                   [18, 19, 20, 21, 22, 23], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]])
    ],
    [
        {'pixdim': (2., 5., 6.), 'mode': 'constant'},
        np.arange(24).reshape((1, 4, 6)),  # data
        {'original_affine': np.array([[-4, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1]]), 'interp_order': 0},
        np.array([[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17],
                   [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [18, 19, 20, 21, 22, 23]]])
    ],
]


class TestSpacingCase(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_spacing(self, init_param, img, data_param, expected_output):
        res = Spacing(**init_param)(img, **data_param)
        np.testing.assert_allclose(res[0], expected_output)
        if 'original_affine' in data_param:
            np.testing.assert_allclose(res[1], data_param['original_affine'])
        np.testing.assert_allclose(init_param['pixdim'],
                                   np.sqrt(np.sum(np.square(res[2]), axis=0))[:len(init_param['pixdim'])])


if __name__ == '__main__':
    unittest.main()
