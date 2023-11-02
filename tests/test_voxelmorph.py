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
from monai.networks.nets import VoxelMorph
from tests.utils import test_script_save

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASE_0 = [  # single channel 3D, batch 1, non-diffeomorphic
    # i.e., VoxelMorph as it is in the original paper
    # https://arxiv.org/pdf/1809.05231.pdf
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "integration_steps": 0,
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_1 = [  # single channel 3D, batch 1, diffeomorphic (default)
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_2 = [  # single channel 3D, batch 1, diffeomorphic, integration at half resolution
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "integration_steps": 7,
        "half_res": True,
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_3 = [  # single channel 3D, batch 1, diffeomorphic, integration at half resolution,
    # using strided convolution for downsampling instead of maxpooling
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "integration_steps": 7,
        "half_res": True,
        "use_maxpool": False,
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_4 = [  # single channel 3D, batch 1, diffeomorphic, integration at half resolution,
    # using strided convolution for downsampling instead of maxpooling,
    # explicitly specify leakyrelu with a different negative slope for final convolutions
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "final_conv_act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "integration_steps": 7,
        "half_res": True,
        "use_maxpool": False,
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_5 = [  # single channel 3D, batch 1, diffeomorphic, integration at half resolution,
    # using strided convolution for downsampling instead of maxpooling,
    # explicitly specify leakyrelu with a different negative slope for both unet and final convolutions.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "final_conv_act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "act": ("leakyrelu", {"negative_slope": 0.1, "inplace": True}),
        "integration_steps": 7,
        "half_res": True,
        "use_maxpool": False,
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_6 = [  # 2-channel 3D, batch 1, diffeomorphic
    # i.e., possible use case where the input contains both modalities (e.g., T1 and T2)
    {
        "spatial_dims": 3,
        "in_channels": 4,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    ((1, 2, 96, 96, 48), (1, 2, 96, 96, 48)),
    ((1, 2, 96, 96, 48), (1, 3, 96, 96, 48)),
]

TEST_CASE_7 = [  # single channel 3D, batch 2, diffeomorphic
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    ((2, 1, 96, 96, 48), (2, 1, 96, 96, 48)),
    ((2, 1, 96, 96, 48), (2, 3, 96, 96, 48)),
]

TEST_CASE_8 = [  # single channel 2D, batch 2, diffeomorphic
    {
        "spatial_dims": 2,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    ((2, 1, 96, 96), (2, 1, 96, 96)),
    ((2, 1, 96, 96), (2, 2, 96, 96)),
]

TEST_CASE_9 = [  # single channel 3D, batch 2, diffeomorphic,
    # one additional level in the UNet with 32 channels in both down and up branch.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    },
    ((2, 1, 96, 96, 48), (2, 1, 96, 96, 48)),
    ((2, 1, 96, 96, 48), (2, 3, 96, 96, 48)),
]

TEST_CASE_10 = [  # single channel 3D, batch 2, diffeomorphic,
    # one additional level in the UNet with 32 channels in both down and up branch.
    # and removed one of the two final convolution blocks.
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32, 32, 32),
        "final_conv_channels": (16,),
    },
    ((2, 1, 96, 96, 48), (2, 1, 96, 96, 48)),
    ((2, 1, 96, 96, 48), (2, 3, 96, 96, 48)),
]

TEST_CASE_11 = [  # single channel 3D, batch 1, diffeomorphic,
    # only one level in the UNet
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32),
        "final_conv_channels": (16, 16),
    },
    ((1, 1, 96, 96, 48), (1, 1, 96, 96, 48)),
    ((1, 1, 96, 96, 48), (1, 3, 96, 96, 48)),
]

CASES = [
    TEST_CASE_0,
    TEST_CASE_1,
    TEST_CASE_2,
    TEST_CASE_3,
    TEST_CASE_4,
    TEST_CASE_5,
    TEST_CASE_6,
    TEST_CASE_7,
    TEST_CASE_8,
    TEST_CASE_9,
    TEST_CASE_10,
    TEST_CASE_11,
]

ILL_CASE_0 = [  # spatial_dims = 1
    {
        "spatial_dims": 1,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_1 = [  # in_channels = 3 (not divisible by 2)
    {
        "spatial_dims": 3,
        "in_channels": 3,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_2 = [  # len(channels) = 0
    {"spatial_dims": 3, "in_channels": 2, "unet_out_channels": 32, "channels": (), "final_conv_channels": (16, 16)}
]

ILL_CASE_3 = [  # channels not in pairs
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
    }
]

ILL_CASE_4 = [  # len(kernel_size) = 3, spatial_dims = 2
    {
        "spatial_dims": 2,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "kernel_size": (3, 3, 3),
    }
]

ILL_CASE_5 = [  # len(up_kernel_size) = 2, spatial_dims = 3
    {
        "spatial_dims": 3,
        "in_channels": 2,
        "unet_out_channels": 32,
        "channels": (16, 32, 32, 32, 32, 32),
        "final_conv_channels": (16, 16),
        "up_kernel_size": (3, 3),
    }
]

ILL_CASES = [ILL_CASE_0, ILL_CASE_1, ILL_CASE_2, ILL_CASE_3, ILL_CASE_4, ILL_CASE_5]


class TestVOXELMORPH(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = VoxelMorph(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape[0]).to(device), torch.randn(input_shape[1]).to(device))
            self.assertEqual(result[0].shape, expected_shape[0])
            self.assertEqual(result[1].shape, expected_shape[1])

    def test_script(self):
        net = VoxelMorph(
            spatial_dims=3,
            in_channels=2,
            unet_out_channels=32,
            channels=(16, 32, 32, 32, 32, 32),
            final_conv_channels=(16, 16),
        ).net
        test_data = torch.randn(1, 2, 96, 96, 48)
        test_script_save(net, test_data)

    @parameterized.expand(ILL_CASES)
    def test_ill_input_hyper_params(self, input_param):
        with self.assertRaises(ValueError):
            _ = VoxelMorph(**input_param)


if __name__ == "__main__":
    unittest.main()
