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

from monai.networks import eval_mode, train_mode
from monai.networks.nets import STUNET_CONFIGS, STUNet, create_stunet
from tests.test_utils import dict_product

device = "cuda" if torch.cuda.is_available() else "cpu"

POOL_OP_KERNEL_SIZES = ((2, 2, 2),) * 4 + ((1, 1, 2),)

TEST_CASE_SHAPE = [
    [params, (2, params["in_channels"], 64, 64, 64), (2, params["out_channels"], 64, 64, 64)]
    for params in dict_product(
        in_channels=[1, 2],
        out_channels=[2, 3],
        deep_supervision=[False, True],
        dims=[(8, 16, 32, 64, 128, 128), (8, 16, 32, 32, 32, 32)],
    )
]

TEST_CASE_DS_SCALES = [
    [  # default pooling: full, /2, /4, /8, /16 in every axis
        {"dims": (8, 16, 32, 64, 128, 128)},
        (1, 1, 64, 64, 64),
        [(1, 2, 64, 64, 64), (1, 2, 32, 32, 32), (1, 2, 16, 16, 16), (1, 2, 8, 8, 8), (1, 2, 4, 4, 4)],
    ],
    [  # anisotropic pooling [[1,2,2],[2,2,2],[2,2,2],[2,2,2],[1,1,2]]: bottleneck at x/8, y/16, z/32
        {
            "dims": (8, 16, 32, 64, 128, 128),
            "pool_op_kernel_sizes": ((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 2)),
        },
        (1, 1, 64, 64, 64),
        [(1, 2, 64, 64, 64), (1, 2, 64, 32, 32), (1, 2, 32, 16, 16), (1, 2, 16, 8, 8), (1, 2, 8, 4, 4)],
    ],
]

TEST_CASE_INVALID = [
    [{"dims": (8, 16, 32)}, "dims"],  # pool_op_kernel_sizes has 5 entries
    [{"conv_kernel_sizes": ((3, 3, 3),) * 5}, "conv_kernel_sizes"],
    [{"depth": (1, 1)}, "depth"],
    [{"pool_op_kernel_sizes": ((2, 2),) * 5}, "3 elements"],
    [{"dims": (8,), "depth": (1,), "pool_op_kernel_sizes": (), "conv_kernel_sizes": ((3, 3, 3),)}, "at least one"],
    [{"depth": (1, 1, 1, 1, 1, 0)}, ">= 1"],
]


class TestSTUNet(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SHAPE)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = STUNet(pool_op_kernel_sizes=POOL_OP_KERNEL_SIZES, **input_param).to(device)
        with eval_mode(net):
            result = net(torch.randn(input_shape).to(device))
            self.assertEqual(tuple(result.shape), expected_shape)

    @parameterized.expand(TEST_CASE_DS_SCALES)
    def test_train_deep_supervision_scales(self, input_param, input_shape, expected_shapes):
        net = STUNet(in_channels=1, out_channels=2, deep_supervision=True, **input_param).to(device)
        with train_mode(net):
            result = net(torch.randn(input_shape).to(device))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), len(expected_shapes))
        for r, e in zip(result, expected_shapes):
            self.assertEqual(tuple(r.shape), e)

    def test_train_deep_supervision_off(self):
        net = STUNet(in_channels=1, out_channels=2, dims=(8, 16, 32, 64, 128, 128), deep_supervision=False).to(device)
        with train_mode(net):
            result = net(torch.randn(1, 1, 64, 64, 64).to(device))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(tuple(result.shape), (1, 2, 64, 64, 64))

    @parameterized.expand([["S"], ["B"], ["L"], ["H"]])
    def test_create_stunet(self, variant):
        with torch.device("meta"):  # build large presets without allocating memory
            net = create_stunet(variant=variant, in_channels=1, out_channels=3)
        self.assertEqual(net.pool_op_kernel_sizes, [list(k) for k in POOL_OP_KERNEL_SIZES])
        dims, depth = STUNET_CONFIGS[variant.upper()]["dims"], STUNET_CONFIGS[variant.upper()]["depth"]
        self.assertEqual(len(net.conv_blocks_context), len(dims))
        self.assertEqual(len(net.conv_blocks_context[-1]), depth[-1])

    def test_create_stunet_invalid_variant(self):
        with self.assertRaises(ValueError):
            create_stunet(variant="X")

    def test_param_counts(self):
        # official model zoo values (in_channels=1, out_channels=105, i.e. the released
        # TotalSegmentator models with 104 structures + background):
        # S 14.6M, B 58.26M, L 440.30M, H 1457.33M -- https://github.com/uni-medical/STU-Net
        expected = {"S": 14_596_941, "B": 58_262_669, "L": 440_302_605, "H": 1_457_326_221}
        for variant, count in expected.items():
            with self.subTest(variant=variant):
                with torch.device("meta"):  # build large presets without allocating memory
                    net = create_stunet(variant=variant, in_channels=1, out_channels=105)
                self.assertEqual(sum(p.numel() for p in net.parameters()), count)

    @parameterized.expand(TEST_CASE_INVALID)
    def test_invalid_config(self, kwargs, message_part):
        with self.assertRaises(ValueError) as cm:
            STUNet(**kwargs)
        self.assertIn(message_part, str(cm.exception))

    def test_divisibility_error(self):
        net = STUNet(in_channels=1, out_channels=2, dims=(8, 16, 32, 64, 128, 128)).to(device)
        with eval_mode(net):
            with self.assertRaises(ValueError) as cm:
                net(torch.randn(1, 1, 60, 60, 60).to(device))
            self.assertIn("divisible", str(cm.exception))

    def test_dimension_error(self):
        net = STUNet(in_channels=1, out_channels=2, dims=(8, 16, 32, 64, 128, 128)).to(device)
        with eval_mode(net):
            with self.assertRaises(ValueError) as cm:
                net(torch.randn(1, 1, 64, 64).to(device))
            self.assertIn("5D", str(cm.exception))

    def test_backward(self):
        net = STUNet(in_channels=1, out_channels=2, dims=(8, 16, 32, 64, 128, 128)).to(device)
        with train_mode(net):
            outputs = net(torch.randn(1, 1, 64, 64, 64).to(device))
            loss = sum(o.sum() for o in outputs)
            loss.backward()
        first_weight = net.conv_blocks_context[0][0].conv1.weight
        self.assertIsNotNone(first_weight.grad)
        self.assertTrue(torch.isfinite(first_weight.grad).all())

    def test_state_dict_layout(self):
        # sub-module names follow the reference implementation so released checkpoints load without key remapping
        net = STUNet(in_channels=1, out_channels=3, dims=(8, 16, 32, 64, 128, 128))
        keys = set(net.state_dict())
        for prefix, count in (
            ("conv_blocks_context", 6),
            ("conv_blocks_localization", 5),
            ("upsample_layers", 5),
            ("seg_outputs", 5),
        ):
            stage_keys = {k.split(".")[1] for k in keys if k.startswith(prefix)}
            self.assertEqual(len(stage_keys), count, prefix)
        self.assertIn("conv_blocks_context.0.0.conv1.weight", keys)
        self.assertIn("conv_blocks_context.0.0.conv3.weight", keys)  # 1x1x1 shortcut of the stem
        self.assertIn("conv_blocks_context.1.0.conv1.weight", keys)
        self.assertIn("upsample_layers.0.conv.weight", keys)
        self.assertIn("conv_blocks_localization.0.0.conv1.weight", keys)
        self.assertIn("seg_outputs.4.weight", keys)  # highest resolution head
        self.assertEqual(net.state_dict()["seg_outputs.4.weight"].shape, (3, 8, 1, 1, 1))
        self.assertEqual(net.state_dict()["upsample_layers.0.conv.weight"].shape, (128, 128, 1, 1, 1))
        self.assertEqual(net.state_dict()["conv_blocks_context.5.0.conv1.weight"].shape, (128, 128, 3, 3, 3))


if __name__ == "__main__":
    unittest.main()
