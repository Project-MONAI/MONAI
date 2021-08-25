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
        dict(as_tensor_output=True, device=None, spatial_size=(2, 2, 2), cache_grid=True),
        {"img": torch.ones((1, 3, 3, 3))},
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
            as_tensor_output=True,
            padding_mode="zeros",
            spatial_size=(2, 2, 2),
            cache_grid=True,
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
    [
        dict(
            prob=0.9,
            rotate_range=(np.pi / 2,),
            shear_range=[1, 2],
            translate_range=[2, 1],
            scale_range=[0.1, 0.2],
            spatial_size=(3, 3),
            cache_grid=True,
            as_tensor_output=True,
            device=None,
        ),
        {"img": torch.arange(64).reshape((1, 8, 8))},
        torch.tensor([[[18.7362, 15.5820, 12.4278], [27.3988, 24.2446, 21.0904], [36.0614, 32.9072, 29.7530]]]),
    ],
]

ARR_NUMPY = np.arange(9 * 10).reshape(1, 9, 10)
ARR_TORCH = torch.Tensor(ARR_NUMPY)
TEST_CASES_SKIPPED_CONSISTENCY = []
for im in (ARR_NUMPY, ARR_TORCH):
    for as_tensor_output in (True, False):
        for in_dtype_is_int in (True, False):
            TEST_CASES_SKIPPED_CONSISTENCY.append((im, as_tensor_output, in_dtype_is_int))


class TestRandAffine(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_rand_affine(self, input_param, input_data, expected_val):
        g = RandAffine(**input_param)
        g.set_random_state(123)
        result = g(**input_data)
        if input_param.get("cache_grid", False):
            self.assertTrue(g._cached_grid is not None)
        self.assertEqual(isinstance(result, torch.Tensor), isinstance(expected_val, torch.Tensor))
        if isinstance(result, torch.Tensor):
            np.testing.assert_allclose(result.cpu().numpy(), expected_val.cpu().numpy(), rtol=1e-4, atol=1e-4)
        else:
            np.testing.assert_allclose(result, expected_val, rtol=1e-4, atol=1e-4)

    def test_ill_cache(self):
        with self.assertWarns(UserWarning):
            RandAffine(cache_grid=True)
        with self.assertWarns(UserWarning):
            RandAffine(cache_grid=True, spatial_size=(1, 1, -1))

    @parameterized.expand(TEST_CASES_SKIPPED_CONSISTENCY)
    def test_skipped_transform_consistency(self, im, as_tensor_output, in_dtype_is_int):
        t1 = RandAffine(prob=0, as_tensor_output=as_tensor_output)
        t2 = RandAffine(prob=1, spatial_size=(10, 11), as_tensor_output=as_tensor_output)

        # change dtype to int32 or float32
        if in_dtype_is_int:
            im = im.astype("int32") if isinstance(im, np.ndarray) else im.int()
        else:
            im = im.astype("float32") if isinstance(im, np.ndarray) else im.float()

        out1 = t1(im)
        out2 = t2(im)

        # check same type
        self.assertEqual(type(out1), type(out2))
        # check matching dtype
        self.assertEqual(out1.dtype, out2.dtype)


if __name__ == "__main__":
    unittest.main()
