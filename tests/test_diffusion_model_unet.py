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
from monai.networks.nets import DiffusionModelUNet
from tests.utils import test_script_save

UNCOND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": (1, 1, 2),
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, True, True),
            "num_head_channels": (0, 2, 4),
            "norm_num_groups": 8,
        }
    ],
]

UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": (0, 0, 4),
            "norm_num_groups": 8,
        }
    ],
]

COND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "upcast_attention": True,
        }
    ],
]

DROPOUT_OK = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "dropout_cattn": 0.25,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        }
    ],
]

DROPOUT_WRONG = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "dropout_cattn": 3.0,
        }
    ]
]


class TestDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16))

    def test_timestep_with_wrong_shape(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1, 1)).long())

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 16, 16))

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                num_channels=(8, 8, 12),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    def test_attention_levels_with_different_length_num_head_channels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                num_channels=(8, 8, 8),
                attention_levels=(False, False, False),
                num_head_channels=(0, 2),
                norm_num_groups=8,
            )

    def test_num_res_blocks_with_different_length_num_channels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(1, 1),
                num_channels=(8, 8, 8),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
            norm_num_groups=8,
            num_head_channels=8,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    def test_with_conditioning_cross_attention_dim_none(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                num_channels=(8, 8, 8),
                attention_levels=(False, False, True),
                with_conditioning=True,
                transformer_num_layers=1,
                cross_attention_dim=None,
                norm_num_groups=8,
            )

    def test_context_with_conditioning_none(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=False,
            transformer_num_layers=1,
            norm_num_groups=8,
        )

        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(
                    x=torch.rand((1, 1, 16, 32)),
                    timesteps=torch.randint(0, 1000, (1,)).long(),
                    context=torch.rand((1, 1, 3)),
                )

    def test_shape_conditioned_models_class_conditioning(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                class_labels=torch.randint(0, 2, (1,)).long(),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    def test_conditioned_models_no_class_labels(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )

        with self.assertRaises(ValueError):
            net.forward(x=torch.rand((1, 1, 16, 32)), timesteps=torch.randint(0, 1000, (1,)).long())

    def test_model_num_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                num_channels=(8, 8, 8),
                attention_levels=(False, False),
                norm_num_groups=8,
                num_head_channels=8,
                num_class_embeds=2,
            )

    def test_script_unconditioned_2d_models(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
        )
        test_script_save(net, torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long())

    def test_script_conditioned_2d_models(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
        )
        test_script_save(net, torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 1, 3)))

    @parameterized.expand(COND_CASES_2D)
    def test_conditioned_2d_models_shape(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 1, 3)))
            self.assertEqual(result.shape, (1, 1, 16, 16))


class TestDiffusionModelUNet3D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_3D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=4,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 16, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 16, 16, 16))

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(16, 16, 16),
            attention_levels=(False, False, True),
            norm_num_groups=16,
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 16, 16)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    def test_script_unconditioned_3d_models(self):
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
        )
        test_script_save(net, torch.rand((1, 1, 16, 16, 16)), torch.randint(0, 1000, (1,)).long())

    def test_script_conditioned_3d_models(self):
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            num_channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
        )
        test_script_save(
            net, torch.rand((1, 1, 16, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 1, 3))
        )

    # Test dropout specification for cross-attention blocks
    @parameterized.expand(DROPOUT_WRONG)
    def test_wrong_dropout(self, input_param):
        with self.assertRaises(ValueError):
            _ = DiffusionModelUNet(**input_param)

    @parameterized.expand(DROPOUT_OK)
    def test_right_dropout(self, input_param):
        _ = DiffusionModelUNet(**input_param)


if __name__ == "__main__":
    unittest.main()
