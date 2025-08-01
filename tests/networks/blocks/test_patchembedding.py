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
from unittest import skipUnless

import torch
import torch.nn as nn
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.patchembedding import PatchEmbed, PatchEmbeddingBlock
from monai.utils import optional_import
from tests.test_utils import SkipIfBeforePyTorchVersion, dict_product

einops, has_einops = optional_import("einops")


TEST_CASE_PATCHEMBEDDINGBLOCK = [
    [
        params,
        (2, params["in_channels"], *([params["img_size"]] * params["spatial_dims"])),
        (2, (params["img_size"] // params["patch_size"]) ** params["spatial_dims"], params["hidden_size"]),
    ]
    for params in dict_product(
        dropout_rate=[0.5],
        in_channels=[1, 4],
        hidden_size=[96, 288],
        img_size=[32, 64],
        patch_size=[8, 16],
        num_heads=[8, 12],
        proj_type=["conv", "perceptron"],
        pos_embed_type=["none", "learnable", "sincos"],
        spatial_dims=[2, 3],
    )
]

img_size = 96
TEST_CASE_PATCHEMBED = [
    [
        params,
        (2, params["in_chans"], *([img_size] * params["spatial_dims"])),
        (2, params["embed_dim"], *([img_size // params["patch_size"]]) * params["spatial_dims"]),
    ]
    for params in dict_product(
        patch_size=[2], in_chans=[1, 4], embed_dim=[6, 12], norm_layer=[nn.LayerNorm], spatial_dims=[2, 3]
    )
]


@SkipIfBeforePyTorchVersion((1, 11, 1))
class TestPatchEmbeddingBlock(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @parameterized.expand(TEST_CASE_PATCHEMBEDDINGBLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = PatchEmbeddingBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_sincos_pos_embed(self):
        net = PatchEmbeddingBlock(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            hidden_size=96,
            num_heads=8,
            pos_embed_type="sincos",
            dropout_rate=0.5,
        )

        self.assertEqual(net.position_embeddings.requires_grad, False)

    def test_learnable_pos_embed(self):
        net = PatchEmbeddingBlock(
            in_channels=1,
            img_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            hidden_size=96,
            num_heads=8,
            pos_embed_type="learnable",
            dropout_rate=0.5,
        )

        self.assertEqual(net.position_embeddings.requires_grad, True)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(128, 128, 128),
                patch_size=(16, 16, 16),
                hidden_size=128,
                num_heads=12,
                proj_type="conv",
                pos_embed_type="sincos",
                dropout_rate=5.0,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(32, 32, 32),
                patch_size=(64, 64, 64),
                hidden_size=512,
                num_heads=8,
                proj_type="perceptron",
                pos_embed_type="sincos",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(96, 96, 96),
                patch_size=(8, 8, 8),
                hidden_size=512,
                num_heads=14,
                proj_type="conv",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                num_heads=8,
                proj_type="perceptron",
                dropout_rate=0.3,
            )
        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=1,
                img_size=(97, 97, 97),
                patch_size=(4, 4, 4),
                hidden_size=768,
                num_heads=8,
                proj_type="perceptron",
                dropout_rate=0.3,
            )

        with self.assertRaises(ValueError):
            PatchEmbeddingBlock(
                in_channels=4,
                img_size=(96, 96, 96),
                patch_size=(16, 16, 16),
                hidden_size=768,
                num_heads=12,
                proj_type="perc",
                dropout_rate=0.3,
            )


class TestPatchEmbed(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(4)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @parameterized.expand(TEST_CASE_PATCHEMBED)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = PatchEmbed(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            PatchEmbed(patch_size=(2, 2, 2), in_chans=1, embed_dim=24, norm_layer=nn.LayerNorm, spatial_dims=5)


if __name__ == "__main__":
    unittest.main()
