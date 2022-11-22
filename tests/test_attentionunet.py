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
from tests.utils import skip_if_no_cuda


class TestAttentionUnet(unittest.TestCase):
    def test_attention_block(self):
        for dims in [2, 3]:
            block = att.AttentionBlock(dims, f_int=2, f_g=6, f_l=6)
            shape = (4, 6) + (30,) * dims
            x = torch.rand(*shape, dtype=torch.float32)
            output = block(x, x)
            self.assertEqual(output.shape, x.shape)

            block = att.AttentionBlock(dims, f_int=2, f_g=3, f_l=6)
            xshape = (4, 6) + (30,) * dims
            x = torch.rand(*xshape, dtype=torch.float32)
            gshape = (4, 3) + (30,) * dims
            g = torch.rand(*gshape, dtype=torch.float32)
            output = block(g, x)
            self.assertEqual(output.shape, x.shape)

    def test_attentionunet(self):
        for dims in [2, 3]:
            shape = (3, 1) + (92,) * dims
            input = torch.rand(*shape)
            model = att.AttentionUnet(
                spatial_dims=dims, in_channels=1, out_channels=2, channels=(3, 4, 5), up_kernel_size=5, strides=(1, 2)
            )
            output = model(input)
            self.assertEqual(output.shape[2:], input.shape[2:])
            self.assertEqual(output.shape[0], input.shape[0])
            self.assertEqual(output.shape[1], 2)

    @skip_if_no_cuda
    def test_attentionunet_gpu(self):
        for dims in [2, 3]:
            shape = (3, 1) + (92,) * dims
            input = torch.rand(*shape).to("cuda:0")
            model = att.AttentionUnet(
                spatial_dims=dims, in_channels=1, out_channels=2, channels=(3, 4, 5), strides=(2, 2)
            ).to("cuda:0")
            with torch.no_grad():
                output = model(input)
                self.assertEqual(output.shape[2:], input.shape[2:])
                self.assertEqual(output.shape[0], input.shape[0])
                self.assertEqual(output.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
