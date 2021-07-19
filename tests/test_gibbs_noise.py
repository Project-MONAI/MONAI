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
from copy import deepcopy

import numpy as np
import torch
from parameterized import parameterized

from monai.data.synthetic import create_test_image_2d, create_test_image_3d
from monai.transforms import GibbsNoise
from monai.utils.misc import set_determinism

TEST_CASES = []
for shape in ((128, 64), (64, 48, 80)):
    for as_tensor_output in (True, False):
        for as_tensor_input in (True, False):
            TEST_CASES.append((shape, as_tensor_output, as_tensor_input))


class TestGibbsNoise(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, as_tensor_input):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        im = create_test_image(*im_shape, num_objs=4, rad_max=20, noise_max=0.0, num_seg_classes=5)[0][None]
        return torch.Tensor(im) if as_tensor_input else im

    @parameterized.expand(TEST_CASES)
    def test_same_result(self, im_shape, as_tensor_output, as_tensor_input):
        im = self.get_data(im_shape, as_tensor_input)
        alpha = 0.8
        t = GibbsNoise(alpha, as_tensor_output)
        out1 = t(deepcopy(im))
        out2 = t(deepcopy(im))
        np.testing.assert_allclose(out1, out2)
        self.assertIsInstance(out1, torch.Tensor if as_tensor_output else np.ndarray)

    @parameterized.expand(TEST_CASES)
    def test_identity(self, im_shape, _, as_tensor_input):
        im = self.get_data(im_shape, as_tensor_input)
        alpha = 0.0
        t = GibbsNoise(alpha)
        out = t(deepcopy(im))
        np.testing.assert_allclose(im, out, atol=1e-2)

    @parameterized.expand(TEST_CASES)
    def test_alpha_1(self, im_shape, _, as_tensor_input):
        im = self.get_data(im_shape, as_tensor_input)
        alpha = 1.0
        t = GibbsNoise(alpha)
        out = t(deepcopy(im))
        np.testing.assert_allclose(0 * im, out)


if __name__ == "__main__":
    unittest.main()
