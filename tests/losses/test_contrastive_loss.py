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

import math
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.losses import ContrastiveLoss

TEST_CASES = [
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {"input": torch.tensor([[1.0, 1.0, 0.0, 0.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
    [  # shape: (2, 4), (2, 4)
        {"temperature": 0.5},
        {
            "input": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
            "target": torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        1.0986,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {
            "input": torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 0.0, 0.0]]),
            "target": torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        },
        0.8719,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.5},
        {"input": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
    [  # shape: (1, 4), (1, 4)
        {"temperature": 0.05},
        {"input": torch.tensor([[0.0, 0.0, 1.0, 1.0]]), "target": torch.tensor([[1.0, 1.0, 0.0, 0.0]])},
        0.0,
    ],
]


class TestContrastiveLoss(unittest.TestCase):

    @parameterized.expand(
        [(temperature, dtype) for temperature in (0.5, 0.01, 0.001) for dtype in (torch.float16, torch.bfloat16)]
    )
    def test_reduced_precision_loss_and_gradients(self, temperature, dtype):
        # All nonself similarities are 0.5, but the embedding gradients are nonzero.
        vectors = torch.cat([torch.ones(4, 1), torch.eye(4)], dim=1).to(dtype)
        inputs = vectors[:2].clone().requires_grad_()
        targets = vectors[2:].clone().requires_grad_()
        ref_inputs = inputs.detach().double().requires_grad_()
        ref_targets = targets.detach().double().requires_grad_()
        representations = torch.nn.functional.normalize(torch.cat([ref_inputs, ref_targets]), dim=1)
        logits = (representations @ representations.T / temperature).masked_fill(
            torch.eye(4, dtype=torch.bool), -torch.inf
        )
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 3, 0, 1]))
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result.double(), expected, atol=1e-5, rtol=1e-5)
        result.backward()
        expected.backward()
        torch.testing.assert_close(inputs.grad, ref_inputs.grad.to(dtype), atol=1e-4, rtol=1e-2)
        torch.testing.assert_close(targets.grad, ref_targets.grad.to(dtype), atol=1e-4, rtol=1e-2)

    @parameterized.expand(
        [(1e-7, torch.float32), (1e-8, torch.float32), (1e-12, torch.float64), (1e-50, torch.float64)]
    )
    def test_small_temperature_cancellation(self, temperature, dtype):
        inputs = torch.ones((2, 4), dtype=dtype, requires_grad=True)
        targets = torch.ones_like(inputs, requires_grad=True)
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        torch.testing.assert_close(result, result.new_tensor(math.log(3)), atol=1e-7, rtol=1e-7)
        result.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(targets.grad).all())

    def test_float16_batch_reduction(self):
        # The mean is finite although a float16 sum of the per-anchor losses overflows.
        batch_size, temperature = 1280, 0.1
        inputs = torch.tensor([[1.0, 0.0]], dtype=torch.float16).repeat(batch_size, 1)
        result = ContrastiveLoss(temperature=temperature)(inputs, -inputs)
        expected = 2 / temperature + math.log(batch_size - 1 + batch_size * math.exp(-2 / temperature))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, result.new_tensor(expected))

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_cpu_autocast(self, dtype):
        vectors = torch.cat([torch.ones(4, 1), torch.eye(4)], dim=1).to(dtype)
        inputs = vectors[:2].clone().requires_grad_()
        targets = vectors[2:].clone().requires_grad_()
        with torch.autocast(device_type="cpu", dtype=dtype):
            result = ContrastiveLoss(temperature=0.001)(inputs, targets)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, result.new_tensor(math.log(3)))
        result.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(targets.grad).all())

    def test_gradcheck(self):
        inputs = torch.tensor([[1.0, 0.2, -0.3], [-0.1, 0.7, 0.5]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0.8, -0.4, 0.1], [0.3, 0.6, -0.2]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(ContrastiveLoss(temperature=0.1), (inputs, targets)))

    @parameterized.expand(
        [(temperature, dtype) for temperature in (0.01, 0.001) for dtype in (torch.float32, torch.float64)]
    )
    def test_identical_low_temperature(self, temperature, dtype):
        inputs = torch.ones((2, 4), dtype=dtype, requires_grad=True)
        targets = torch.ones((2, 4), dtype=dtype, requires_grad=True)
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        # Every anchor has three equally similar candidates, including its positive.
        torch.testing.assert_close(result, result.new_tensor(math.log(3)), atol=1e-4, rtol=1e-4)
        result.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(targets.grad).all())

    @parameterized.expand(
        [(temperature, dtype) for temperature in (0.01, 0.001) for dtype in (torch.float32, torch.float64)]
    )
    def test_antipodal_low_temperature(self, temperature, dtype):
        inputs = torch.eye(2, dtype=dtype, requires_grad=True)
        targets = (-torch.eye(2, dtype=dtype)).requires_grad_()
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        # Positive similarity is -1, and both other candidates have similarity 0.
        expected = 1 / temperature + math.log(2 + math.exp(-1 / temperature))
        torch.testing.assert_close(result, result.new_tensor(expected))
        result.backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertTrue(torch.isfinite(targets.grad).all())

    @parameterized.expand(
        [(temperature, dtype) for temperature in (0.5, 0.05, 0.01) for dtype in (torch.float32, torch.float64)]
    )
    def test_reference_loss_and_gradients(self, temperature, dtype):
        inputs = torch.tensor([[1.0, 0.2, -0.3], [-0.1, 0.7, 0.5]], dtype=dtype, requires_grad=True)
        targets = torch.tensor([[0.8, -0.4, 0.1], [0.3, 0.6, -0.2]], dtype=dtype, requires_grad=True)
        reference_inputs = inputs.detach().double().requires_grad_()
        reference_targets = targets.detach().double().requires_grad_()
        representations = torch.nn.functional.normalize(torch.cat([reference_inputs, reference_targets]), dim=1)
        logits = representations @ representations.T / temperature
        logits = logits.masked_fill(torch.eye(4, dtype=torch.bool), -torch.inf)
        # The positive for each anchor is the corresponding sample in the other view.
        expected = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 3, 0, 1]))
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        torch.testing.assert_close(result.double(), expected, atol=1e-4, rtol=1e-4)
        result.backward()
        expected.backward()
        torch.testing.assert_close(inputs.grad.double(), reference_inputs.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(targets.grad.double(), reference_targets.grad, atol=1e-4, rtol=1e-4)

    @parameterized.expand([(temperature, sign) for temperature in (0.01, 0.001) for sign in (1, -1)])
    def test_single_pair_low_temperature(self, temperature, sign):
        inputs = torch.tensor([[1.0, 0.0]], requires_grad=True)
        targets = (sign * inputs.detach()).requires_grad_()
        result = ContrastiveLoss(temperature=temperature)(inputs, targets)
        torch.testing.assert_close(result, torch.zeros_like(result))
        result.backward()
        torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))
        torch.testing.assert_close(targets.grad, torch.zeros_like(targets))

    @parameterized.expand(TEST_CASES)
    def test_result(self, input_param, input_data, expected_val):
        contrastiveloss = ContrastiveLoss(**input_param)
        result = contrastiveloss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = ContrastiveLoss(temperature=0.5)
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((1, 2, 3)), torch.ones((1, 1, 2, 3)))

    def test_with_cuda(self):
        loss = ContrastiveLoss(temperature=0.5)
        i = torch.ones((1, 10))
        j = torch.ones((1, 10))
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 0.0, atol=1e-4, rtol=1e-4)

    def check_warning_rasied(self):
        with self.assertWarns(Warning):
            ContrastiveLoss(temperature=0.5, batch_size=1)


if __name__ == "__main__":
    unittest.main()
