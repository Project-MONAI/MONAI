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

from monai.networks.layers import EMAQuantizer, VectorQuantizer


class TestEMA(unittest.TestCase):
    def test_ema_shape(self):
        layer = EMAQuantizer(spatial_dims=2, num_embeddings=16, embedding_dim=8)
        input_shape = (1, 8, 8, 8)
        x = torch.randn(input_shape)
        layer = layer.train()
        outputs = layer(x)
        self.assertEqual(outputs[0].shape, input_shape)
        self.assertEqual(outputs[2].shape, (1, 8, 8))
 
        layer = layer.eval()
        outputs = layer(x)
        self.assertEqual(outputs[0].shape, input_shape)
        self.assertEqual(outputs[2].shape, (1, 8, 8))

    def test_ema_quantize(self):
        layer = EMAQuantizer(spatial_dims=2, num_embeddings=16, embedding_dim=8)
        input_shape = (1, 8, 8, 8)
        x = torch.randn(input_shape)
        outputs = layer.quantize(x)
        self.assertEqual(outputs[0].shape, (64, 8))  # (HxW, C)
        self.assertEqual(outputs[1].shape, (64, 16))  # (HxW, E)
        self.assertEqual(outputs[2].shape, (1, 8, 8))  # (1, H, W)

    def test_ema(self):
        layer = EMAQuantizer(spatial_dims=2, num_embeddings=2, embedding_dim=2, epsilon=0, decay=0)
        original_weight_0 = layer.embedding.weight[0].clone()
        original_weight_1 = layer.embedding.weight[1].clone()
        x_0 = original_weight_0
        x_0 = x_0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_0 = x_0.repeat(1, 1, 1, 2) + 0.001

        x_1 = original_weight_1
        x_1 = x_1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x_1 = x_1.repeat(1, 1, 1, 2)

        x = torch.cat([x_0, x_1], dim=0)
        layer = layer.train()
        _ = layer(x)

        self.assertTrue(all(layer.embedding.weight[0] != original_weight_0))
        self.assertTrue(all(layer.embedding.weight[1] == original_weight_1))


class TestVectorQuantizer(unittest.TestCase):
    def test_vector_quantizer_shape(self):
        layer = VectorQuantizer(EMAQuantizer(spatial_dims=2, num_embeddings=16, embedding_dim=8))
        input_shape = (1, 8, 8, 8)
        x = torch.randn(input_shape)
        outputs = layer(x)
        self.assertEqual(outputs[1].shape, input_shape)

    def test_vector_quantizer_quantize(self):
        layer = VectorQuantizer(EMAQuantizer(spatial_dims=2, num_embeddings=16, embedding_dim=8))
        input_shape = (1, 8, 8, 8)
        x = torch.randn(input_shape)
        outputs = layer.quantize(x)
        self.assertEqual(outputs.shape, (1, 8, 8))


if __name__ == "__main__":
    unittest.main()
