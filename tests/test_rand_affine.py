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

import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import RandAffine

TEST_CASES = [
    [
        dict(as_tensor_output=False, device=None),
        {"img": torch.arange(27).reshape((3, 3, 3))},
        np.arange(27).reshape((3, 3, 3)),
    ],
    [
        dict(as_tensor_output=False, device=None, spatial_size=-1),
        {"img": torch.arange(27).reshape((3, 3, 3))},
        np.arange(27).reshape((3, 3, 3)),
    ],
    [
        dict(as_tensor_output=False, device=None),
        {"img": torch.arange(27).reshape((3, 3, 3)), "spatial_size": (2, 2)},
        np.array([[[2.0, 3.0], [5.0, 6.0]], [[11.0, 12.0], [14.0, 15.0]], [[20.0, 21.0], [23.0, 24.0]]]),
    ],
    [
        dict(as_tensor_output=True, device=None),
        {"img": torch.ones((1, 3, 3, 3)), "spatial_size": (2, 2, 2)},
        torch.ones((1, 2, 2, 2)),
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            as_tensor_output=True,
            padding_mode="zeros",
            spatial_size=(2, 2, 2),
            device=None,
        ),
        {"img": torch.ones((1, 3, 3, 3)), "mode": "bilinear"},
        torch.tensor([[[[0.3658, 1.0000], [1.0000, 1.0000]], [[1.0000, 1.0000], [1.0000, 0.9333]]]]),
    ],
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            as_tensor_output=True,
            device=None,
        ),
        {"img": torch.arange(64).reshape((1, 8, 8)), "spatial_size": (3, 3)},
        torch.tensor([[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]),
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


if __name__ == "__main__":
    unittest.main()
