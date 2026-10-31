# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import tempfile
import unittest

import torch

from monai.bundle import inspect_ckpt


class TestInspectCkpt(unittest.TestCase):
    def setUp(self):
        # Create a temporary checkpoint file with a simple state dict
        self.tmp_dir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.tmp_dir, "model.pt")
        state_dict = {
            "layer1.weight": torch.randn(4, 3),
            "layer1.bias": torch.zeros(4),
            "layer2.weight": torch.randn(2, 4),
        }
        torch.save(state_dict, self.ckpt_path)

    def test_returns_dict_with_correct_keys(self):
        result = inspect_ckpt(path=self.ckpt_path, print_all_vars=False)
        self.assertIsInstance(result, dict)
        self.assertIn("layer1.weight", result)
        self.assertIn("layer1.bias", result)
        self.assertIn("layer2.weight", result)

    def test_shapes_are_correct(self):
        result = inspect_ckpt(path=self.ckpt_path, print_all_vars=False)
        self.assertEqual(result["layer1.weight"]["shape"], (4, 3))
        self.assertEqual(result["layer1.bias"]["shape"], (4,))
        self.assertEqual(result["layer2.weight"]["shape"], (2, 4))

    def test_dtype_is_reported(self):
        result = inspect_ckpt(path=self.ckpt_path, print_all_vars=False)
        self.assertIn("dtype", result["layer1.weight"])
        self.assertTrue(result["layer1.weight"]["dtype"].startswith("torch."))

    def test_compute_hash_md5(self):
        # Should not raise; hash value is logged but not returned in dict
        result = inspect_ckpt(path=self.ckpt_path, print_all_vars=False, compute_hash=True, hash_type="md5")
        self.assertIsInstance(result, dict)

    def test_compute_hash_sha256(self):
        result = inspect_ckpt(path=self.ckpt_path, print_all_vars=False, compute_hash=True, hash_type="sha256")
        self.assertIsInstance(result, dict)

    def test_print_all_vars_true_does_not_raise(self):
        # Should log each variable without raising
        try:
            inspect_ckpt(path=self.ckpt_path, print_all_vars=True)
        except Exception as e:
            self.fail(f"inspect_ckpt raised an exception with print_all_vars=True: {e}")


if __name__ == "__main__":
    unittest.main()
