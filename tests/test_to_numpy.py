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

from monai.transforms import ToNumpy


class TestToNumpy(unittest.TestCase):
    def test_numpy_input(self):
        test_data = np.array([[1, 2], [3, 4]])
        test_data = np.rot90(test_data)
        self.assertFalse(test_data.flags["C_CONTIGUOUS"])
        result = ToNumpy()(test_data)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(result, test_data)

    def test_tensor_input(self):
        test_data = torch.tensor([[1, 2], [3, 4]])
        test_data = test_data.rot90()
        self.assertFalse(test_data.is_contiguous())
        result = ToNumpy()(test_data)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        np.testing.assert_allclose(result, test_data.numpy())

    def test_list_tuple(self):
        test_data = [[1, 2], [3, 4]]
        result = ToNumpy()(test_data)
        np.testing.assert_allclose(result, np.asarray(test_data))
        test_data = ((1, 2), (3, 4))
        result = ToNumpy()(test_data)
        np.testing.assert_allclose(result, np.asarray(test_data))


if __name__ == "__main__":
    unittest.main()
