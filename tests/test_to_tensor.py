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

from monai.transforms import ToTensor


class TestToTensor(unittest.TestCase):
    def test_array_input(self):
        for test_data in ([[1, 2], [3, 4]], np.array([[1, 2], [3, 4]]), torch.as_tensor([[1, 2], [3, 4]])):
            result = ToTensor()(test_data)
            torch.testing.assert_allclose(result, test_data)
            self.assertTupleEqual(result.shape, (2, 2))

    def test_single_input(self):
        for test_data in (5, np.asarray(5), torch.tensor(5)):
            result = ToTensor()(test_data)
            torch.testing.assert_allclose(result, test_data)
            self.assertEqual(result.ndim, 0)


if __name__ == "__main__":
    unittest.main()
