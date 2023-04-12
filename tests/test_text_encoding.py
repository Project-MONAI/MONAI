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
from monai.networks.blocks.text_embedding import TextEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestTextEncoder(unittest.TestCase):
    def test_test_encoding_shape(self):
        # test 2D encoder
        text_encoder = TextEncoder(spatial_dims=2, out_channels=32, encoding="clip_encoding_univeral_model_32", pretrained=True).to(device)
        text_encoding = text_encoder()
        print(text_encoding.shape)
        self.assertEqual(text_encoding.shape, (32,256,1,1))

        # test 3D encoder
        text_encoder = TextEncoder(spatial_dims=3, out_channels=32, encoding="clip_encoding_univeral_model_32", pretrained=True).to(device)
        text_encoding = text_encoder()
        print(text_encoding.shape)
        self.assertEqual(text_encoding.shape, (32,256,1,1,1))

        # test random enbedding 
        text_encoder = TextEncoder(spatial_dims=3, out_channels=32, encoding="rand_embedding", pretrained=True).to(device)
        text_encoding = text_encoder()
        print(text_encoding.shape)
        self.assertEqual(text_encoding.shape, (32,256,1,1,1))

if __name__ == "__main__":
    unittest.main()
