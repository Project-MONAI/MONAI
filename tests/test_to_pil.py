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
from typing import TYPE_CHECKING
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.transforms import ToPIL
from monai.utils import optional_import

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageImage
    from PIL.Image import fromarray as PILImage_fromarray

    has_pil = True
else:
    PILImage_fromarray, has_pil = optional_import("PIL.Image", name="fromarray")
    PILImageImage, _ = optional_import("PIL.Image", name="Image")

TEST_CASE_ARRAY_1 = [np.array([[1.0, 2.0], [3.0, 4.0]])]
TEST_CASE_TENSOR_1 = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]


class TestToPIL(unittest.TestCase):
    @parameterized.expand([TEST_CASE_ARRAY_1])
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_numpy_input(self, test_data):
        self.assertTrue(isinstance(test_data, np.ndarray))
        result = ToPIL()(test_data)
        self.assertTrue(isinstance(result, PILImageImage))
        np.testing.assert_allclose(np.array(result), test_data)

    @parameterized.expand([TEST_CASE_TENSOR_1])
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_tensor_input(self, test_data):
        self.assertTrue(isinstance(test_data, torch.Tensor))
        result = ToPIL()(test_data)
        self.assertTrue(isinstance(result, PILImageImage))
        np.testing.assert_allclose(np.array(result), test_data.numpy())

    @parameterized.expand([TEST_CASE_ARRAY_1])
    @skipUnless(has_pil, "Requires `pillow` package.")
    def test_pil_input(self, test_data):
        test_data_pil = PILImage_fromarray(test_data)
        self.assertTrue(isinstance(test_data_pil, PILImageImage))
        result = ToPIL()(test_data_pil)
        self.assertTrue(isinstance(result, PILImageImage))
        np.testing.assert_allclose(np.array(result), test_data)


if __name__ == "__main__":
    unittest.main()
