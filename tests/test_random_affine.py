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

from monai.transforms.transforms import RandAffine

TEST_CASES = [
    [
        dict(as_tensor_output=False, device=None), {'img': torch.ones((3, 3, 3)), 'spatial_size': (2, 2)},
        np.ones((3, 2, 2))
    ],
    [
        dict(as_tensor_output=True, device=None), {'img': torch.ones((1, 3, 3, 3)), 'spatial_size': (2, 2, 2)},
        torch.ones((1, 2, 2, 2))
    ],
    [
        dict(prob=0.9,
             rotate_range=(np.pi / 2,),
             shear_range=[1, 2],
             translate_range=[2, 1],
             as_tensor_output=True,
             spatial_size=(2, 2, 2),
             device=None), {'img': torch.ones((1, 3, 3, 3)), 'mode': 'bilinear'},
        torch.tensor([[[[1.0000, 0.7776], [0.4174, 0.0780]], [[0.0835, 1.0000], [0.3026, 0.5732]]]],)
    ],
    [
        dict(prob=0.9,
             rotate_range=(np.pi / 2,),
             shear_range=[1, 2],
             translate_range=[2, 1],
             scale_range=[.1, .2],
             as_tensor_output=True,
             device=None), {'img': torch.arange(64).reshape((1, 8, 8)), 'spatial_size': (3, 3)},
        torch.tensor([[[27.3614, 18.0237, 8.6860], [40.0440, 30.7063, 21.3686], [52.7266, 43.3889, 34.0512]]])
    ],
]


class TestRandAffine(unittest.TestCase):

    @parameterized.expand(TEST_CASES)
    def test_rand_affine(self, input_param, input_data, expected_val):
        g = RandAffine(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        self.assertEqual(torch.is_tensor(result), torch.is_tensor(expected_val))
        if torch.is_tensor(result):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
