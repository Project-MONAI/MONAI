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
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.layers import Act, Norm
from monai.networks.nets.unet import CheckpointUNet, UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # single channel 2D, batch 16, no residual
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

TEST_CASE_1 = [  # single channel 2D, batch 16
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

TEST_CASE_2 = [  # single channel 3D, batch 16
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

TEST_CASE_3 = [  # 4-channel 3D, batch 16
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

TEST_CASE_4 = [  # 4-channel 3D, batch 16, batch normalization
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "norm": Norm.BATCH,
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_5 = [  # 4-channel 3D, batch 16, LeakyReLU activation
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (Act.LEAKYRELU, {"negative_slope": 0.2}),
        "adn_ordering": "NA",
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

TEST_CASE_6 = [  # 4-channel 3D, batch 16, LeakyReLU activation explicit
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 3,
        "channels": (16, 32, 64),
        "strides": (2, 2),
        "num_res_units": 1,
        "act": (torch.nn.LeakyReLU, {"negative_slope": 0.2}),
    },
    (16, 4, 32, 64, 48),
    (16, 3, 32, 64, 48),
]

CASES = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6]


class TestCheckpointUNet(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        """Validate CheckpointUNet output shapes across configurations.

        Args:
            input_param: Dictionary of UNet constructor arguments.
            input_shape: Tuple specifying input tensor dimensions.
            expected_shape: Tuple specifying expected output tensor dimensions.
        """
        net = CheckpointUNet(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_checkpointing_equivalence_eval(self):
        """Confirm eval parity when checkpointing is inactive."""
        params = dict(
            spatial_dims=2, in_channels=1, out_channels=2, channels=(8, 16, 32), strides=(2, 2), num_res_units=1
        )

        x = torch.randn(2, 1, 32, 32, device=device)

        torch.manual_seed(42)
        net_plain = UNet(**params).to(device)

        torch.manual_seed(42)
        net_ckpt = CheckpointUNet(**params).to(device)

        # Both in eval mode disables checkpointing logic
        with eval_mode(net_ckpt), eval_mode(net_plain):
            y_ckpt = net_ckpt(x)
            y_plain = net_plain(x)

        # Check shape equality
        self.assertEqual(y_ckpt.shape, y_plain.shape)

        # Check numerical equivalence
        self.assertTrue(
            torch.allclose(y_ckpt, y_plain, atol=1e-6, rtol=1e-5),
            f"Eval-mode outputs differ: max abs diff={torch.max(torch.abs(y_ckpt - y_plain)).item():.2e}",
        )

    def test_checkpointing_activates_training(self):
        """Verify checkpointing recomputes activations during training."""
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


if __name__ == "__main__":
    unittest.main()
