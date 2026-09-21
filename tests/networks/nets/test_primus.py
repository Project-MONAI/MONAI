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

import json
import os
import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import Primus, PrimusS, convert_primus_state_dict, create_primus
from monai.utils import optional_import
from tests.test_utils import dict_product

dna_primus, has_dna = optional_import("dynamic_network_architectures.architectures.primus")

TESTS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# upstream PrimusV3 state dict layout for this config, see "source" in the json file
UPSTREAM_STATE_DICT = os.path.join(TESTS_PATH, "testing_data", "primus_v3_state_dict.json")
UPSTREAM_CONFIG = {
    "in_channels": 1,
    "out_channels": 2,
    "img_size": 16,
    "embed_dim": 48,
    "num_layers": 2,
    "num_heads": 2,
    "num_register_tokens": 2,
    "channels_per_level": (4, 8, 8, 16),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# small transformer so the tests stay fast; head_dim = 24 is divisible by 2 * spatial_dims for 2D and 3D
SMALL = {"embed_dim": 48, "num_layers": 2, "num_heads": 2, "channels_per_level": (4, 8, 8, 16)}

TEST_CASE_PRIMUS = [
    [
        {**SMALL, **params, "in_channels": 2, "out_channels": 3, "img_size": (16, 24, 32)[: params["spatial_dims"]]},
        (2, 2, *(16, 24, 32)[: params["spatial_dims"]]),
        (2, 3, *(16, 24, 32)[: params["spatial_dims"]]),
    ]
    for params in dict_product(
        spatial_dims=[2, 3],
        num_register_tokens=[0, 2],
        use_rope=[True, False],
        use_abs_pos_embed=[True, False],
        add_skips=[True, False],
    )
]

# outputs of upstream PrimusV3 for UPSTREAM_CONFIG with deterministic_weights and deterministic_input (float64)
UPSTREAM_TOKENS = [
    0.585765320134539,
    -0.4004680850654125,
    0.18435013358440758,
    0.44993196022974735,
    0.386034922766048,
    -0.3517999822712953,
    0.2974185279001581,
    -0.23754968692199674,
]
UPSTREAM_OUTPUT = [
    -0.35732740945891495,
    -0.38404543372147304,
    -0.4494280946488171,
    -0.3842308424323807,
    -0.3326295288644062,
    -0.685464194443321,
    -0.666793117906307,
    -0.6244707660537548,
]


def deterministic_weights(state_dict):
    """Fan-in scaled sinusoidal values for every entry, seeded by the sorted key order."""
    order = {k: i for i, k in enumerate(sorted(state_dict))}
    weights = {}
    for k, v in state_dict.items():
        w = torch.sin(torch.arange(v.numel(), dtype=torch.float64) * 0.37 + order[k]).reshape(v.shape)
        is_matrix = v.ndim > 1 and v.shape[0] > 1 and v.shape[1:].numel() > 1
        weights[k] = w / v[0].numel() ** 0.5 if is_matrix else 0.5 * w
    return weights


class TestPrimus(unittest.TestCase):
    @parameterized.expand(TEST_CASE_PRIMUS)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = Primus(**input_param).to(device)
        with eval_mode(net):
            result, mask = net(torch.randn(input_shape).to(device), return_mask=True)
        self.assertEqual(result.shape, expected_shape)
        self.assertIsNone(mask)

    def test_patch_drop(self):
        net = Primus(1, 2, 32, patch_drop_rate=0.75, num_register_tokens=2, **SMALL).to(device)
        net.train()
        x = torch.randn(2, 1, 32, 32, 32, device=device)
        out, mask = net(x, return_mask=True)
        self.assertEqual(out.shape, (2, 2, 32, 32, 32))
        self.assertEqual(mask.shape, (2, 1, 32, 32, 32))
        self.assertEqual(mask.dtype, torch.bool)
        # 64 tokens of 8^3 voxels, 16 of them kept per sample
        self.assertEqual(mask.flatten(1).sum(1).tolist(), [16 * 8**3] * 2)
        # the mask is constant within each patch
        patches = mask.view(2, 4, 8, 4, 8, 4, 8)
        self.assertTrue(torch.equal(patches.amin((2, 4, 6)), patches.amax((2, 4, 6))))
        out.sum().backward()
        self.assertIsNotNone(net.register_tokens.grad)
        with eval_mode(net):
            _, mask = net(x, return_mask=True)
        self.assertIsNone(mask)

    def test_depth_per_level(self):
        net = Primus(1, 2, (8, 16, 4), depth_per_level=(2, 1), **{**SMALL, "channels_per_level": (4, 8, 16)})
        self.assertEqual(net.patch_size, (4, 4, 4))
        self.assertEqual(net.grid_size, (2, 4, 1))
        with eval_mode(net):
            self.assertEqual(net(torch.randn(1, 1, 8, 16, 4)).shape, (1, 2, 8, 16, 4))

    def test_stem_norm(self):
        net = Primus(1, 2, 16, stem_norm=("group", {"num_groups": 2}), **SMALL)
        self.assertIsInstance(net.down_projection.stem.norm1, torch.nn.GroupNorm)
        with eval_mode(net):
            self.assertEqual(net(torch.randn(1, 1, 16, 16, 16)).shape, (1, 2, 16, 16, 16))

    def test_variant(self):
        net = PrimusS(in_channels=1, out_channels=2, img_size=16)
        self.assertEqual(len(net.blocks), 12)
        self.assertEqual(net.norm.normalized_shape, (396,))
        with self.assertRaises(ValueError):
            create_primus("XL", in_channels=1, out_channels=2, img_size=16)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            Primus(1, 2, 20, **SMALL)  # not divisible by the patch size
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, **{**SMALL, "num_heads": 5})
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, **{**SMALL, "num_heads": 0})
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, **{**SMALL, "num_heads": 3})  # rope needs head_dim 16 divisible by 2 * spatial_dims
        with self.assertRaises(ValueError):
            Primus(1, 2, 16, patch_drop_rate=1.0, **SMALL)
        bad_args = [
            {"spatial_dims": 4},
            {"spatial_dims": 1},
            {"depth_per_level": (), "channels_per_level": (4,)},
            {"num_register_tokens": -1},
            {"drop_path_rate": -0.1},
            {"attention_dropout_rate": 1.5},
            {"dropout_rate": -1.0},
        ]
        for args in bad_args:
            with self.subTest(**args), self.assertRaises(ValueError):
                Primus(1, 2, 16, **{**SMALL, **args})
        net = Primus(1, 2, 16, **SMALL)
        with self.assertRaises(ValueError):
            net(torch.randn(1, 1, 32, 32, 32))

    def test_convert_state_dict_keys(self):
        old = {
            "eva.blocks.0.attn.q_proj.weight": 0,
            "eva.pos_embed": 1,
            "down_projection.stem.blocks.0.conv1.conv.weight": 2,
            "down_projection.stem.blocks.0.conv1.all_modules.0.weight": 2,
            "down_projection.stem.blocks.0.conv2.norm.bias": 3,
            "down_projection.stem.blocks.0.skip.0.conv.weight": 4,
            "down_projection.stem.blocks.0.skip.0.norm.weight": 5,
            "down_projection.stages.1.blocks.2.skip.1.norm.weight": 6,
            "up_projection.decode.0.1.weight": 7,
        }
        expected = {
            "blocks.0.attn.q_proj.weight": 0,
            "pos_embed": 1,
            "down_projection.stem.conv1.weight": 2,
            "down_projection.stem.norm2.bias": 3,
            "down_projection.stem.skip.conv.weight": 4,
            "down_projection.stem.skip.norm.weight": 5,
            "down_projection.stages.1.2.skip.norm.weight": 6,
            "up_projection.decode.0.1.weight": 7,
        }
        self.assertEqual(convert_primus_state_dict(old), expected)

    def test_convert_upstream_layout(self):
        # every upstream key maps to a MONAI key of the same shape and vice versa (strict load)
        with open(UPSTREAM_STATE_DICT) as f:
            shapes = json.load(f)["shapes"]
        net = Primus(**UPSTREAM_CONFIG)
        net.load_old_state_dict({k: torch.zeros(v) for k, v in shapes.items()})

    def test_upstream_golden_values(self):
        net = Primus(**UPSTREAM_CONFIG).double()
        net.load_state_dict(deterministic_weights(net.state_dict()))
        x = torch.sin(torch.arange(16**3, dtype=torch.float64) * 0.11).reshape(1, 1, 16, 16, 16)
        tokens = []
        net.norm.register_forward_hook(lambda module, inputs, output: tokens.append(output))
        with eval_mode(net), torch.no_grad():
            out = net(x)
        # transformer tokens are sensitive to details the decoder damps (e.g. LayerNorm eps), so check both
        expected_tokens = torch.tensor(UPSTREAM_TOKENS, dtype=torch.float64)
        expected_output = torch.tensor(UPSTREAM_OUTPUT, dtype=torch.float64)
        torch.testing.assert_close(
            tokens[0].flatten()[torch.arange(8) * 59 + 7], expected_tokens, rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(out.flatten()[torch.arange(8) * 997 + 13], expected_output, rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(has_dna, "Requires dynamic-network-architectures.")
    def test_load_old_state_dict(self):
        old = dna_primus.PrimusV3(
            1, 48, (8, 8, 8), 2, eva_depth=2, eva_numheads=2, input_shape=(16, 16, 16), num_register_tokens=2
        )
        net = Primus(1, 2, 16, num_register_tokens=2, **SMALL | {"channels_per_level": (32, 64, 256, 1024)})
        net.load_old_state_dict(old.state_dict())
        x = torch.randn(1, 1, 16, 16, 16)
        with eval_mode(old, net):
            torch.testing.assert_close(net(x), old(x))


if __name__ == "__main__":
    unittest.main()
