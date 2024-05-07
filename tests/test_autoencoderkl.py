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

import os
import tempfile
import unittest

import torch
from parameterized import parameterized

from monai.apps import download_url
from monai.networks import eval_mode
from monai.networks.nets import AutoencoderKL
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion, skip_if_downloading_fails, testing_data_config

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
_, has_einops = optional_import("einops")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


CASES_NO_ATTENTION = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
]

CASES_ATTENTION = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": (1, 1, 2),
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, False),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
            "with_encoder_nonlocal_attn": False,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16),
        (1, 1, 16, 16),
        (1, 4, 4, 4),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4, 4),
            "latent_channels": 4,
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
]

if has_einops:
    CASES = CASES_NO_ATTENTION + CASES_ATTENTION
else:
    CASES = CASES_NO_ATTENTION


class TestAutoEncoderKL(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape, expected_latent_shape):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    @parameterized.expand(CASES)
    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_with_convtranspose_and_checkpointing(
        self, input_param, input_shape, expected_shape, expected_latent_shape
    ):
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=1,
                norm_num_groups=16,
            )

    def test_model_num_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False),
                latent_channels=8,
                num_res_blocks=1,
                norm_num_groups=16,
            )

    def test_model_num_channels_not_same_size_of_num_res_blocks(self):
        with self.assertRaises(ValueError):
            AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(24, 24, 24),
                attention_levels=(False, False, False),
                latent_channels=8,
                num_res_blocks=(8, 8),
                norm_num_groups=16,
            )

    def test_shape_reconstruction(self):
        input_param, input_shape, expected_shape, _ = CASES[0]
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.reconstruct(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_reconstruction_with_convtranspose_and_checkpointing(self):
        input_param, input_shape, expected_shape, _ = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.reconstruct(torch.randn(input_shape).to(device))
            self.assertEqual(result.shape, expected_shape)

    def test_shape_encode(self):
        input_param, input_shape, _, expected_latent_shape = CASES[0]
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.encode(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_latent_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_encode_with_convtranspose_and_checkpointing(self):
        input_param, input_shape, _, expected_latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.encode(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_latent_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)

    def test_shape_sampling(self):
        input_param, _, _, expected_latent_shape = CASES[0]
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.sampling(
                torch.randn(expected_latent_shape).to(device), torch.randn(expected_latent_shape).to(device)
            )
            self.assertEqual(result.shape, expected_latent_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_sampling_convtranspose_and_checkpointing(self):
        input_param, _, _, expected_latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.sampling(
                torch.randn(expected_latent_shape).to(device), torch.randn(expected_latent_shape).to(device)
            )
            self.assertEqual(result.shape, expected_latent_shape)

    def test_shape_decode(self):
        input_param, expected_input_shape, _, latent_shape = CASES[0]
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)

    @SkipIfBeforePyTorchVersion((1, 11))
    def test_shape_decode_convtranspose_and_checkpointing(self):
        input_param, expected_input_shape, _, latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)

    def test_compatibility_with_monai_generative(self):
        with skip_if_downloading_fails():
            net = AutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                channels=(4, 4, 4),
                latent_channels=4,
                attention_levels=(False, False, True),
                num_res_blocks=1,
                norm_num_groups=4,
            ).to(device)

            tmpdir = tempfile.mkdtemp()
            key = "autoencoderkl_monai_generative_weights"
            url = testing_data_config("models", key, "url")
            hash_type = testing_data_config("models", key, "hash_type")
            hash_val = testing_data_config("models", key, "hash_val")
            filename = "autoencoderkl_monai_generative_weights.pt"

            weight_path = os.path.join(tmpdir, filename)
            download_url(url=url, filepath=weight_path, hash_val=hash_val, hash_type=hash_type)

            net.load_old_state_dict(torch.load(weight_path), verbose=False)

            expected = torch.Tensor(
                [
                    [
                        [
                            [-0.1196, 0.1790, 0.3983, 0.0833, -0.5382, -0.1774, 0.5868, -0.1472],
                            [-0.1426, -0.0725, -0.1253, -0.2386, -0.1912, -0.6041, -0.0240, -0.4589],
                            [-0.3699, 0.1424, 0.1764, 0.1204, -0.4929, -0.9932, -0.1486, -0.4411],
                            [-0.6254, 0.0475, 0.1803, 0.4053, 0.3762, -1.0841, 0.5859, -0.0620],
                            [-0.5583, -0.0289, -0.2902, -0.3155, 0.6079, -1.0878, 0.8552, 0.3166],
                            [-0.3613, 0.0557, -0.1652, -0.3117, 0.7810, -0.3432, 1.0887, 0.5672],
                            [0.4715, 0.0455, 0.1007, -0.8551, 0.1393, -0.1390, 1.3083, 0.6711],
                            [0.2841, -0.0138, -0.1377, -0.1850, 0.2664, 0.0295, 0.9022, 0.7321],
                        ]
                    ]
                ]
            ).to(device)
            with eval_mode(net):
                # fix random state
                torch.manual_seed(0)
                result, _, _ = net.forward(torch.randn((1, 1, 8, 8)).to(device))

                torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
