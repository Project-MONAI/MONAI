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

import unittest

import torch

from monai.losses import AUCMLoss
from tests.test_utils import test_script_save


class TestAUCMLoss(unittest.TestCase):
    def test_v1(self):
        loss_fn = AUCMLoss(version="v1")
        input = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()
        loss = loss_fn(input, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_v2(self):
        loss_fn = AUCMLoss(version="v2")
        input = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()
        loss = loss_fn(input, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_invalid_version(self):
        with self.assertRaises(ValueError):
            AUCMLoss(version="invalid")

    def test_invalid_input_shape(self):
        loss_fn = AUCMLoss()
        input = torch.randn(32, 2)  # Wrong channel
        target = torch.randint(0, 2, (32, 1)).float()
        with self.assertRaises(ValueError):
            loss_fn(input, target)

    def test_invalid_target_shape(self):
        loss_fn = AUCMLoss()
        input = torch.randn(32, 1)
        target = torch.randint(0, 2, (32, 2)).float()  # Wrong channel
        with self.assertRaises(ValueError):
            loss_fn(input, target)

    def test_shape_mismatch(self):
        loss_fn = AUCMLoss()
        input = torch.randn(32, 1)
        target = torch.randint(0, 2, (16, 1)).float()
        with self.assertRaises(ValueError):
            loss_fn(input, target)

    def test_backward(self):
        loss_fn = AUCMLoss()
        input = torch.randn(32, 1, requires_grad=True)
        target = torch.randint(0, 2, (32, 1)).float()
        loss = loss_fn(input, target)
        loss.backward()
        self.assertIsNotNone(input.grad)

    def test_script_save(self):
        loss_fn = AUCMLoss()
        test_script_save(loss_fn, torch.randn(32, 1), torch.randint(0, 2, (32, 1)).float())


if __name__ == "__main__":
    unittest.main()
