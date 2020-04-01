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

from monai.transforms import RandAffineGrid

TEST_CASES = [
    [{'as_tensor_output': False, 'device': None}, {'grid': torch.ones((3, 3, 3))},
     np.ones((3, 3, 3))],
    [{'rotate_range': (1, 2), 'translate_range': (3, 3, 3)}, {'grid': torch.arange(0, 27).reshape((3, 3, 3))},
     torch.tensor(
         np.array([[[-32.81998, -33.910976, -35.001972], [-36.092968, -37.183964, -38.27496],
                    [-39.36596, -40.456955, -41.54795]],
                   [[2.1380205, 3.1015975, 4.0651755], [5.028752, 5.9923296, 6.955907], [7.919484, 8.883063, 9.84664]],
                   [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]]))],
    [{'translate_range': (3, 3, 3), 'as_tensor_output': False, 'device': torch.device('cpu:0')},
     {'spatial_size': (3, 3, 3)},
     np.array([[[[0.17881513, 0.17881513, 0.17881513], [0.17881513, 0.17881513, 0.17881513],
                 [0.17881513, 0.17881513, 0.17881513]],
                [[1.1788151, 1.1788151, 1.1788151], [1.1788151, 1.1788151, 1.1788151],
                 [1.1788151, 1.1788151, 1.1788151]],
                [[2.1788151, 2.1788151, 2.1788151], [2.1788151, 2.1788151, 2.1788151],
                 [2.1788151, 2.1788151, 2.1788151]]],
               [[[-2.283164, -2.283164, -2.283164], [-1.283164, -1.283164, -1.283164],
                 [-0.28316402, -0.28316402, -0.28316402]],
                [[-2.283164, -2.283164, -2.283164], [-1.283164, -1.283164, -1.283164],
                 [-0.28316402, -0.28316402, -0.28316402]],
                [[-2.283164, -2.283164, -2.283164], [-1.283164, -1.283164, -1.283164],
                 [-0.28316402, -0.28316402, -0.28316402]]],
               [[[-2.6388912, -1.6388912, -0.6388912], [-2.6388912, -1.6388912, -0.6388912],
                 [-2.6388912, -1.6388912, -0.6388912]],
                [[-2.6388912, -1.6388912, -0.6388912], [-2.6388912, -1.6388912, -0.6388912],
                 [-2.6388912, -1.6388912, -0.6388912]],
                [[-2.6388912, -1.6388912, -0.6388912], [-2.6388912, -1.6388912, -0.6388912],
                 [-2.6388912, -1.6388912, -0.6388912]]],
               [[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]])],
    [{'rotate_range': (1., 1., 1.), 'shear_range': (0.1,), 'scale_range': (1.2,)},
     {'grid': torch.arange(0, 108).reshape((4, 3, 3, 3))},
     torch.tensor(
         np.array([[[[-9.4201e+00, -8.1672e+00, -6.9143e+00], [-5.6614e+00, -4.4085e+00, -3.1556e+00],
                     [-1.9027e+00, -6.4980e-01, 6.0310e-01]],
                    [[1.8560e+00, 3.1089e+00, 4.3618e+00], [5.6147e+00, 6.8676e+00, 8.1205e+00],
                     [9.3734e+00, 1.0626e+01, 1.1879e+01]],
                    [[1.3132e+01, 1.4385e+01, 1.5638e+01], [1.6891e+01, 1.8144e+01, 1.9397e+01],
                     [2.0650e+01, 2.1902e+01, 2.3155e+01]]],
                   [[[9.9383e-02, -4.8845e-01, -1.0763e+00], [-1.6641e+00, -2.2519e+00, -2.8398e+00],
                     [-3.4276e+00, -4.0154e+00, -4.6032e+00]],
                    [[-5.1911e+00, -5.7789e+00, -6.3667e+00], [-6.9546e+00, -7.5424e+00, -8.1302e+00],
                     [-8.7180e+00, -9.3059e+00, -9.8937e+00]],
                    [[-1.0482e+01, -1.1069e+01, -1.1657e+01], [-1.2245e+01, -1.2833e+01, -1.3421e+01],
                     [-1.4009e+01, -1.4596e+01, -1.5184e+01]]],
                   [[[5.9635e+01, 6.1199e+01, 6.2764e+01], [6.4328e+01, 6.5892e+01, 6.7456e+01],
                     [6.9021e+01, 7.0585e+01, 7.2149e+01]],
                    [[7.3714e+01, 7.5278e+01, 7.6842e+01], [7.8407e+01, 7.9971e+01, 8.1535e+01],
                     [8.3099e+01, 8.4664e+01, 8.6228e+01]],
                    [[8.7792e+01, 8.9357e+01, 9.0921e+01], [9.2485e+01, 9.4049e+01, 9.5614e+01],
                     [9.7178e+01, 9.8742e+01, 1.0031e+02]]],
                   [[[8.1000e+01, 8.2000e+01, 8.3000e+01], [8.4000e+01, 8.5000e+01, 8.6000e+01],
                     [8.7000e+01, 8.8000e+01, 8.9000e+01]],
                    [[9.0000e+01, 9.1000e+01, 9.2000e+01], [9.3000e+01, 9.4000e+01, 9.5000e+01],
                     [9.6000e+01, 9.7000e+01, 9.8000e+01]],
                    [[9.9000e+01, 1.0000e+02, 1.0100e+02], [1.0200e+02, 1.0300e+02, 1.0400e+02],
                     [1.0500e+02, 1.0600e+02, 1.0700e+02]]]]))],
]


class TestRandAffineGrid(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_rand_affine_grid(self, input_param, input_data, expected_val):
        g = RandAffineGrid(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
