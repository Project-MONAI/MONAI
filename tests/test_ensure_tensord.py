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

from monai.transforms import EnsureTensord


class TestEnsureTensord(unittest.TestCase):
    def test_array_input(self):
        for test_data in (np.array([[1, 2], [3, 4]]), torch.as_tensor([[1, 2], [3, 4]])):
            result = EnsureTensord(keys="data")({"data": test_data})["data"]
            self.assertTrue(isinstance(result, torch.Tensor))
            torch.testing.assert_allclose(result, test_data)
            self.assertTupleEqual(result.shape, (2, 2))

    def test_single_input(self):
        for test_data in (5, 5.0, False, np.asarray(5), torch.tensor(5)):
            result = EnsureTensord(keys="data")({"data": test_data})["data"]
            self.assertTrue(isinstance(result, torch.Tensor))
            torch.testing.assert_allclose(result, test_data)
            self.assertEqual(result.ndim, 0)

    def test_string(self):
        # string input
        result = EnsureTensord(keys="data")({"data": "test_string"})["data"]
        self.assertTrue(isinstance(result, str))
        self.assertEqual(result, "test_string")
        # numpy array of string
        result = EnsureTensord(keys="data")({"data": np.array(["test_string0", "test_string1"])})["data"]
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result[1], "test_string1")

    def test_list_tuple(self):
        result = EnsureTensord(keys="data")({"data": [[1, 2], [3, 4]]})["data"]
        self.assertTrue(isinstance(result, list))
        self.assertTrue(isinstance(result[0][1], torch.Tensor))
        torch.testing.assert_allclose(result[1][0], torch.as_tensor(3))
        # tuple of numpy arrays
        result = EnsureTensord(keys="data")({"data": (np.array([1, 2]), np.array([3, 4]))})["data"]
        self.assertTrue(isinstance(result, tuple))
        self.assertTrue(isinstance(result[0], torch.Tensor))
        torch.testing.assert_allclose(result[1], torch.as_tensor([3, 4]))

    def test_dict(self):
        # simulate complicated input data
        test_data = {
            "img": np.array([1.0, 2.0], dtype=np.float32),
            "meta": {"dims": 3, "size": np.array([1, 2, 3]), "path": "temp/test"},
            "extra": None,
        }
        result = EnsureTensord(keys="data")({"data": test_data})["data"]
        self.assertTrue(isinstance(result, dict))
        self.assertTrue(isinstance(result["img"], torch.Tensor))
        torch.testing.assert_allclose(result["img"], torch.as_tensor([1.0, 2.0]))
        self.assertTrue(isinstance(result["meta"]["size"], torch.Tensor))
        torch.testing.assert_allclose(result["meta"]["size"], torch.as_tensor([1, 2, 3]))
        self.assertEqual(result["meta"]["path"], "temp/test")
        self.assertEqual(result["extra"], None)


if __name__ == "__main__":
    unittest.main()
