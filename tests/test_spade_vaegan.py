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

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets import SPADENet

CASE_2D = [[[2, 1, 1, 3, [64, 64], [16, 32, 64, 128], 16, True]]]
CASE_2D_BIS = [[[2, 1, 1, 3, [64, 64], [16, 32, 64, 128], 16, True]]]
CASE_3D = [[[3, 1, 1, 3, [64, 64, 64], [16, 32, 64, 128], 16, True]]]


def create_semantic_data(shape: list, semantic_regions: int):
    """
    To create semantic and image mock inputs for the network.
    Args:
        shape: input shape
        semantic_regions: number of semantic region
    Returns:
    """
    out_label = torch.zeros(shape)
    out_image = torch.zeros(shape) + torch.randn(shape) * 0.01
    for i in range(1, semantic_regions):
        shape_square = [i // np.random.choice(list(range(2, i // 2))) for i in shape]
        start_point = [np.random.choice(list(range(shape[ind] - shape_square[ind]))) for ind, i in enumerate(shape)]
        if len(shape) == 2:
            out_label[
                start_point[0] : (start_point[0] + shape_square[0]), start_point[1] : (start_point[1] + shape_square[1])
            ] = i
            base_intensity = torch.ones(shape_square) * np.random.randn()
            out_image[
                start_point[0] : (start_point[0] + shape_square[0]), start_point[1] : (start_point[1] + shape_square[1])
            ] = (base_intensity + torch.randn(shape_square) * 0.1)
        elif len(shape) == 3:
            out_label[
                start_point[0] : (start_point[0] + shape_square[0]),
                start_point[1] : (start_point[1] + shape_square[1]),
                start_point[2] : (start_point[2] + shape_square[2]),
            ] = i
            base_intensity = torch.ones(shape_square) * np.random.randn()
            out_image[
                start_point[0] : (start_point[0] + shape_square[0]),
                start_point[1] : (start_point[1] + shape_square[1]),
                start_point[2] : (start_point[2] + shape_square[2]),
            ] = (base_intensity + torch.randn(shape_square) * 0.1)
        else:
            ValueError("Supports only 2D and 3D tensors")

    # One hot encode label
    out_label_ = torch.zeros([semantic_regions] + list(out_label.shape))
    for ch in range(semantic_regions):
        out_label_[ch, ...] = out_label == ch

    return out_label_.unsqueeze(0), out_image.unsqueeze(0).unsqueeze(0)


class TestDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(CASE_2D)
    def test_forward_2d(self, input_param):
        """
        Check that forward method is called correctly and output shape matches.
        """
        net = SPADENet(*input_param)
        in_label, in_image = create_semantic_data(input_param[4], input_param[3])
        with eval_mode(net):
            out, z_mu, z_logvar = net(in_label, in_image)
            self.assertTrue(torch.all(torch.isfinite(out)))
            self.assertTrue(torch.all(torch.isfinite(z_mu)))
            self.assertTrue(torch.all(torch.isfinite(z_logvar)))
            self.assertEqual(list(out.shape), [1, 1, 64, 64])

    @parameterized.expand(CASE_2D_BIS)
    def test_encoder_decoder(self, input_param):
        """
        Check that forward method is called correctly and output shape matches.
        """
        net = SPADENet(*input_param)
        in_label, in_image = create_semantic_data(input_param[4], input_param[3])
        with eval_mode(net):
            out_z = net.encode(in_image)
            self.assertEqual(list(out_z.shape), [1, 16])
            out_i = net.decode(in_label, out_z)
            self.assertEqual(list(out_i.shape), [1, 1, 64, 64])

    @parameterized.expand(CASE_3D)
    def test_forward_3d(self, input_param):
        """
        Check that forward method is called correctly and output shape matches.
        """
        net = SPADENet(*input_param)
        in_label, in_image = create_semantic_data(input_param[4], input_param[3])
        with eval_mode(net):
            out, z_mu, z_logvar = net(in_label, in_image)
            self.assertTrue(torch.all(torch.isfinite(out)))
            self.assertTrue(torch.all(torch.isfinite(z_mu)))
            self.assertTrue(torch.all(torch.isfinite(z_logvar)))
            self.assertEqual(list(out.shape), [1, 1, 64, 64, 64])

    def test_shape_wrong(self):
        """
        We input an input shape that isn't divisible by 2**(n downstream steps)
        """
        with self.assertRaises(ValueError):
            _ = SPADENet(1, 1, 8, [16, 16], [16, 32, 64, 128], 16, True)


if __name__ == "__main__":
    unittest.main()
