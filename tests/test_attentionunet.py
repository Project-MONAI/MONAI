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

import unittest

import torch

import monai.networks.nets.attentionunet as att


class TestAttentionUnet(unittest.TestCase):
    def test_attention_block(self):
        for dims in [2, 3]:
            block = att.AttentionBlock(dims, F_int=16, F_g=64, F_l=64)
            shape = (4, 64) + (30,) * dims
            x = torch.rand(*shape, dtype=torch.float32)
            output = block(x, x)
            assert output.shape == x.shape

            block = att.AttentionBlock(dims, F_int=16, F_g=32, F_l=64)
            xshape = (4, 64) + (30,) * dims
            x = torch.rand(*xshape, dtype=torch.float32)
            gshape = (4, 32) + (30,) * dims
            g = torch.rand(*gshape, dtype=torch.float32)
            output = block(g, x)
            assert output.shape == x.shape

    def test_attentionunet(self):
        for dims in [2, 3]:
            shape = (3, 1) + (92,) * dims
            input = torch.rand(*shape)
            model = att.AttentionUnet(
                spatial_dims=dims, in_channels=1, out_channels=2, channels=(16, 32, 64), strides=(2, 2)
            )
            output = model(input)
            assert output.shape[2:] == input.shape[2:]
            assert output.shape[0] == input.shape[0]
            assert output.shape[1] == 2

    def test_attentionunet_gpu(self):
        if torch.cuda.is_available():
            for dims in [2, 3]:
                shape = (3, 1) + (92,) * dims
                input = torch.rand(*shape).to("cuda:0")
                model = att.AttentionUnet(
                    spatial_dims=dims, in_channels=1, out_channels=2, channels=(16, 32, 64), strides=(2, 2)
                ).to("cuda:0")
                with torch.no_grad():
                    output = model(input)
                    assert output.shape[2:] == input.shape[2:]
                    assert output.shape[0] == input.shape[0]
                    assert output.shape[1] == 2


if __name__ == "__main__":
    unittest.main()
