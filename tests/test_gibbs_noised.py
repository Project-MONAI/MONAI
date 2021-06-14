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
from monai.transforms import GibbsNoised
from monai.utils.misc import set_determinism

TEST_CASES = []
for shape in ((128, 64), (64, 48, 80)):
    for as_tensor_output in (True, False):
        for as_tensor_input in (True, False):
            TEST_CASES.append((shape, as_tensor_output, as_tensor_input))

KEYS = ["im", "label"]


class TestGibbsNoised(unittest.TestCase):
    def setUp(self):
        set_determinism(0)
        super().setUp()

    def tearDown(self):
        set_determinism(None)

    @staticmethod
    def get_data(im_shape, as_tensor_input):
        create_test_image = create_test_image_2d if len(im_shape) == 2 else create_test_image_3d
        ims = create_test_image(*im_shape, 4, 20, 0, 5)
        ims = [torch.Tensor(im) for im in ims] if as_tensor_input else ims
        return {k: v for k, v in zip(KEYS, ims)}

    @parameterized.expand(TEST_CASES)
    def test_same_result(self, im_shape, as_tensor_output, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        alpha = 0.8
        t = GibbsNoised(KEYS, alpha, as_tensor_output)
        out1 = t(deepcopy(data))
        out2 = t(deepcopy(data))
        for k in KEYS:
            np.testing.assert_allclose(out1[k], out2[k])
            self.assertIsInstance(out1[k], type(data[k]))

    @parameterized.expand(TEST_CASES)
    def test_identity(self, im_shape, _, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        alpha = 0.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        for k in KEYS:
            np.testing.assert_allclose(data[k], out[k], atol=1e-2)

    @parameterized.expand(TEST_CASES)
    def test_alpha_1(self, im_shape, _, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        alpha = 1.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        for k in KEYS:
            np.testing.assert_allclose(0 * data[k], out[k])

    @parameterized.expand(TEST_CASES)
    def test_dict_matches(self, im_shape, _, as_tensor_input):
        data = self.get_data(im_shape, as_tensor_input)
        data = {KEYS[0]: deepcopy(data[KEYS[0]]), KEYS[1]: deepcopy(data[KEYS[0]])}
        alpha = 1.0
        t = GibbsNoised(KEYS, alpha)
        out = t(deepcopy(data))
        np.testing.assert_allclose(out[KEYS[0]], out[KEYS[1]])


if __name__ == "__main__":
    unittest.main()
