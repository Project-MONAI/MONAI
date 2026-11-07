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

import numpy as np
import torch
import torch.nn as nn
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.crossattention import CrossAttentionBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.utils import optional_import
from tests.test_utils import dict_product

einops, has_einops = optional_import("einops")
TEST_CASE_TRANSFORMERBLOCK = [
    [params, (2, 512, params["hidden_size"]), (2, 512, params["hidden_size"])]
    for params in dict_product(
        dropout_rate=np.linspace(0, 1, 4),
        hidden_size=[360, 480, 600, 768],
        num_heads=[4, 8, 12],
        mlp_dim=[1024, 3072],
        with_cross_attention=[False, True],
    )
]


class TestTransformerBlock(unittest.TestCase):

    @parameterized.expand(TEST_CASE_TRANSFORMERBLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = TransformerBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            TransformerBlock(hidden_size=128, num_heads=12, mlp_dim=2048, dropout_rate=4.0)

        with self.assertRaises(ValueError):
            TransformerBlock(hidden_size=622, num_heads=8, mlp_dim=3072, dropout_rate=0.4)

    @skipUnless(has_einops, "Requires einops")
    def test_cross_attention_is_identity_when_disabled(self):
        block = TransformerBlock(hidden_size=128, mlp_dim=256, num_heads=4, with_cross_attention=False)
        # attributes always exist for typing and checkpoint compatibility
        self.assertTrue(hasattr(block, "cross_attn"))
        self.assertTrue(hasattr(block, "norm_cross_attn"))
        # cross_attn is nn.Identity (no parameters) when disabled
        self.assertIsInstance(block.cross_attn, nn.Identity)
        param_names = [name for name, _ in block.named_parameters()]
        self.assertFalse(any(n.startswith("cross_attn") for n in param_names))

    @skipUnless(has_einops, "Requires einops")
    def test_cross_attention_params_registered_when_enabled(self):
        block = TransformerBlock(hidden_size=128, mlp_dim=256, num_heads=4, with_cross_attention=True)
        self.assertIsInstance(block.cross_attn, CrossAttentionBlock)
        self.assertTrue(hasattr(block, "norm_cross_attn"))
        param_names = [name for name, _ in block.named_parameters()]
        self.assertTrue(any(n.startswith("cross_attn.") for n in param_names))
        self.assertTrue(any("norm_cross_attn" in n for n in param_names))

    @skipUnless(has_einops, "Requires einops")
    def test_cross_attention_forward_with_context(self):
        hidden_size = 128
        block = TransformerBlock(hidden_size=hidden_size, mlp_dim=256, num_heads=4, with_cross_attention=True)
        x = torch.randn(2, 16, hidden_size)
        context = torch.randn(2, 8, hidden_size)
        with eval_mode(block):
            out = block(x, context=context)
        self.assertEqual(out.shape, x.shape)

    @skipUnless(has_einops, "Requires einops")
    def test_access_attn_matrix(self):
        # input format
        hidden_size = 128
        mlp_dim = 12
        num_heads = 2
        dropout_rate = 0
        input_shape = (2, 256, hidden_size)

        # returns an empty attention matrix
        no_matrix_acess_blk = TransformerBlock(
            hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, dropout_rate=dropout_rate
        )
        no_matrix_acess_blk(torch.randn(input_shape))
        assert isinstance(no_matrix_acess_blk.attn.att_mat, torch.Tensor)
        # no of elements is zero
        assert no_matrix_acess_blk.attn.att_mat.nelement() == 0

        # be able to acess the attention matrix
        matrix_acess_blk = TransformerBlock(
            hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, dropout_rate=dropout_rate, save_attn=True
        )
        matrix_acess_blk(torch.randn(input_shape))
        assert matrix_acess_blk.attn.att_mat.shape == (input_shape[0], input_shape[0], input_shape[1], input_shape[1])


if __name__ == "__main__":
    unittest.main()
