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
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import DecoderOnlyTransformer
from monai.utils import optional_import

_, has_einops = optional_import("einops")
TEST_CASES = []
for dropout_rate in np.linspace(0, 1, 2):
    for attention_layer_dim in [360, 480, 600, 768]:
        for num_heads in [4, 6, 8, 12]:
            TEST_CASES.append(
                [
                    {
                        "num_tokens": 10,
                        "max_seq_len": 16,
                        "attn_layers_dim": attention_layer_dim,
                        "attn_layers_depth": 2,
                        "attn_layers_heads": num_heads,
                        "embedding_dropout_rate": dropout_rate,
                    }
                ]
            )


class TestDecoderOnlyTransformer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_unconditioned_models(self, input_param):
        net = DecoderOnlyTransformer(**input_param)
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)))

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_conditioned_models(self, input_param):
        net = DecoderOnlyTransformer(**input_param, with_cross_attention=True)
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)), context=torch.randn(1, 3, input_param["attn_layers_dim"]))

    def test_attention_dim_not_multiple_of_heads(self):
        with self.assertRaises(ValueError):
            DecoderOnlyTransformer(
                num_tokens=10, max_seq_len=16, attn_layers_dim=8, attn_layers_depth=2, attn_layers_heads=3
            )

    @skipUnless(has_einops, "Requires einops")
    def test_dropout_rate_negative(self):

        with self.assertRaises(ValueError):
            DecoderOnlyTransformer(
                num_tokens=10,
                max_seq_len=16,
                attn_layers_dim=8,
                attn_layers_depth=2,
                attn_layers_heads=2,
                embedding_dropout_rate=-1,
            )


if __name__ == "__main__":
    unittest.main()
