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
import torch
from parameterized import parameterized

from monai.transforms.transforms import Rand2DElastic

TEST_CASES = [
    [{'spacing': (.3, .3), 'magnitude_range': (1., 2.), 'prob': 0.0, 'as_tensor_output': False, 'device': None},
     {'img': torch.ones((3, 3, 3)), 'spatial_size': (2, 2)},
     np.ones((3, 2, 2))],
    [
        {'spacing': (.3, .3), 'magnitude_range': (1., 2.), 'prob': 0.9, 'as_tensor_output': False, 'device': None},
        {'img': torch.ones((3, 3, 3)), 'spatial_size': (2, 2), 'mode': 'bilinear'},
        np.array([[[0., 0.608901], [1., 0.5702355]], [[0., 0.608901], [1., 0.5702355]], [[0., 0.608901],
                                                                                         [1., 0.5702355]]]),
    ],
    [
        {
            'spacing': (1., 1.), 'magnitude_range': (1., 1.), 'scale_range': [1.2, 2.2], 'prob': 0.9, 'padding_mode':
            'border', 'as_tensor_output': True, 'device': None, 'spatial_size': (2, 2)
        },
        {'img': torch.arange(27).reshape((3, 3, 3))},
        torch.tensor([[[1.0849, 1.1180], [6.8100, 7.0265]], [[10.0849, 10.1180], [15.8100, 16.0265]],
                      [[19.0849, 19.1180], [24.8100, 25.0265]]]),
    ],
    [
        {
            'spacing': (.3, .3), 'magnitude_range': (1., 2.), 'translate_range': [-.2, .4], 'scale_range': [1.2, 2.2],
            'prob': 0.9, 'as_tensor_output': False, 'device': None
        },
        {'img': torch.arange(27).reshape((3, 3, 3)), 'spatial_size': (2, 2)},
        np.array([[[0., 1.1731534], [3.8834658, 6.0565934]], [[0., 9.907095], [12.883466, 15.056594]],
                  [[0., 18.641037], [21.883465, 24.056593]]]),
    ],
]


class TestRand2DElastic(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_rand_2d_elastic(self, input_param, input_data, expected_val):
        g = Rand2DElastic(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
