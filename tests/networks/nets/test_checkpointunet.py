# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.unet import CheckpointUNet, UNet
from tests.test_utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 0,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_1 = [
    {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 32),
    (16, 3, 32, 32),
]

TEST_CASE_2 = [
    {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 1, 32, 24, 48),
    (16, 3, 32, 24, 48),
]

TEST_CASE_3 = [
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestCheckpointUNet(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = CheckpointUNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_script(self):
        net = CheckpointUNet(
            spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2), num_res_units=0
        )
        test_data = torch.randn(16, 1, 32, 32)
        test_script_save(net, test_data)

    def test_ill_input_shape(self):
        net = CheckpointUNet(spatial_dims=2, in_channels=1, out_channels=3, channels=(16, 32, 64), strides=(2, 2))
        with eval_mode(net):
            with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match"):
                net.forward(torch.randn(2, 1, 16, 5))

    def test_checkpointing_equivalence_eval(self):
        """Ensure that CheckpointUNet matches standard UNet in eval mode (checkpointing inactive)."""
        params = dict(
            spatial_dims=2, in_channels=1, out_channels=2, channels=(8, 16, 32), strides=(2, 2), num_res_units=1
        )

        torch.manual_seed(0)
        x = torch.randn(2, 1, 32, 32, device=device)

        net_plain = UNet(**params).to(device)
        net_ckpt = CheckpointUNet(**params).to(device)
        net_ckpt.load_state_dict(net_plain.state_dict())

        with eval_mode(net_ckpt), eval_mode(net_plain):
            y_ckpt = net_ckpt(x)
            y_plain = net_plain(x)

        # checkpointing should not change outputs in eval mode
        self.assertTrue(torch.allclose(y_ckpt, y_plain, atol=1e-6, rtol=1e-5))

    def test_checkpointing_activates_training(self):
        """Ensure checkpointing triggers recomputation under training and gradients propagate."""
        params = dict(
            spatial_dims=2, in_channels=1, out_channels=1, channels=(8, 16, 32), strides=(2, 2), num_res_units=1
        )

        net = CheckpointUNet(**params).to(device)
        net.train()

        x = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)
        y = net(x)
        loss = y.mean()
        loss.backward()

        # gradient flow check
        grad_norm = sum(p.grad.abs().sum() for p in net.parameters() if p.grad is not None)
        self.assertGreater(grad_norm.item(), 0.0)

        # checkpointing should reduce activation memory use; we can't directly assert memory savings
        # but we can confirm no runtime errors and gradients propagate correctly
        self.assertIsNotNone(grad_norm)


if __name__ == "__main__":
    unittest.main()
