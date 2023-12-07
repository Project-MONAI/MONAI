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

from monai.networks import eval_mode
from monai.networks.nets import DecoderOnlyTransformer


class TestDecoderOnlyTransformer(unittest.TestCase):
    def test_unconditioned_models(self):
        net = DecoderOnlyTransformer(
            num_tokens=10, max_seq_len=16, attn_layers_dim=8, attn_layers_depth=2, attn_layers_heads=2
        )
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)))

    def test_models_with_flash_attention(self):
        net = DecoderOnlyTransformer(
            num_tokens=10,
            max_seq_len=16,
            attn_layers_dim=8,
            attn_layers_depth=2,
            attn_layers_heads=2,
            use_flash_attention=True,
        ).to(torch.device("cuda:0"))
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)).to(torch.device("cuda:0")))

    def test_conditioned_models(self):
        net = DecoderOnlyTransformer(
            num_tokens=10,
            max_seq_len=16,
            attn_layers_dim=8,
            attn_layers_depth=2,
            attn_layers_heads=2,
            with_cross_attention=True,
            embedding_dropout_rate=0,
        )
        with eval_mode(net):
            net.forward(torch.randint(0, 10, (1, 16)), context=torch.randn(1, 4, 8))


if __name__ == "__main__":
    unittest.main()
