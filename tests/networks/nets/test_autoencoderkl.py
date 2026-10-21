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
from unittest import skipUnless

import torch
from parameterized import parameterized

from monai.apps import download_url
from monai.networks import eval_mode
from monai.networks.nets import AutoencoderKL
from monai.utils import optional_import
from tests.test_utils import skip_if_downloading_fails, testing_data_config

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
    _MIGRATION_PARAMS = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (4, 4, 4),
        "latent_channels": 4,
        "attention_levels": (False, False, False),
        "num_res_blocks": 1,
        "norm_num_groups": 4,
    }

    @parameterized.expand(CASES)
    def test_shape(self, input_param, input_shape, expected_shape, expected_latent_shape):
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.forward(torch.randn(input_shape).to(device))
            self.assertEqual(result[0].shape, expected_shape)
            self.assertEqual(result[1].shape, expected_latent_shape)
            self.assertEqual(result[2].shape, expected_latent_shape)

    @parameterized.expand(CASES)
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

    def test_shape_decode_convtranspose_and_checkpointing(self):
        input_param, expected_input_shape, _, latent_shape = CASES[0]
        input_param = input_param.copy()
        input_param.update({"use_checkpoint": True, "use_convtranspose": True})
        net = AutoencoderKL(**input_param).to(device)
        with eval_mode(net):
            result = net.decode(torch.randn(latent_shape).to(device))
            self.assertEqual(result.shape, expected_input_shape)

    @skipUnless(has_einops, "Requires einops")
    def test_compatibility_with_monai_generative(self):
        # test loading weights from a model saved in MONAI Generative, version 0.2.3
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

            net.load_old_state_dict(torch.load(weight_path, weights_only=True), verbose=False)

    @staticmethod
    def _new_to_old_sd(new_sd: dict, include_proj_attn: bool = True) -> dict:
        """Convert new-style state dict keys to legacy naming conventions.

        Args:
            new_sd: State dict with current key naming.
            include_proj_attn: If True, map `.attn.out_proj.` to `.proj_attn.`.

        Returns:
            State dict with legacy key names.
        """
        old_sd: dict = {}
        for k, v in new_sd.items():
            if ".attn.to_q." in k:
                old_sd[k.replace(".attn.to_q.", ".to_q.")] = v.clone()
            elif ".attn.to_k." in k:
                old_sd[k.replace(".attn.to_k.", ".to_k.")] = v.clone()
            elif ".attn.to_v." in k:
                old_sd[k.replace(".attn.to_v.", ".to_v.")] = v.clone()
            elif ".attn.out_proj." in k:
                if include_proj_attn:
                    old_sd[k.replace(".attn.out_proj.", ".proj_attn.")] = v.clone()
            elif "postconv" in k:
                old_sd[k.replace("postconv", "conv")] = v.clone()
            else:
                old_sd[k] = v.clone()
        return old_sd

    @skipUnless(has_einops, "Requires einops")
    def test_load_old_state_dict_proj_attn_copied_to_out_proj(self):
        params = {**self._MIGRATION_PARAMS, "include_fc": True}
        src = AutoencoderKL(**params).to(device)
        old_sd = self._new_to_old_sd(src.state_dict(), include_proj_attn=True)

        # record the tensor values that were stored under proj_attn
        expected = {k.replace(".proj_attn.", ".attn.out_proj."): v for k, v in old_sd.items() if ".proj_attn." in k}
        self.assertGreater(len(expected), 0, "No proj_attn keys in old state dict - check model config")

        dst = AutoencoderKL(**params).to(device)
        dst.load_old_state_dict(old_sd)

        for new_key, expected_val in expected.items():
            torch.testing.assert_close(
                dst.state_dict()[new_key], expected_val.to(device), msg=f"Weight mismatch for {new_key}"
            )

    @skipUnless(has_einops, "Requires einops")
    def test_load_old_state_dict_missing_proj_attn_initialises_identity(self):
        params = {**self._MIGRATION_PARAMS, "include_fc": True}
        src = AutoencoderKL(**params).to(device)
        old_sd = self._new_to_old_sd(src.state_dict(), include_proj_attn=False)

        dst = AutoencoderKL(**params).to(device)
        dst.load_old_state_dict(old_sd)
        loaded = dst.state_dict()

        out_proj_weights = [k for k in loaded if "attn.out_proj.weight" in k]
        out_proj_biases = [k for k in loaded if "attn.out_proj.bias" in k]
        self.assertGreater(len(out_proj_weights), 0, "No out_proj keys found - check model config")

        for k in out_proj_weights:
            n = loaded[k].shape[0]
            torch.testing.assert_close(
                loaded[k], torch.eye(n, dtype=loaded[k].dtype, device=device), msg=f"{k} should be an identity matrix"
            )
        for k in out_proj_biases:
            torch.testing.assert_close(loaded[k], torch.zeros_like(loaded[k]), msg=f"{k} should be all-zeros")

    @skipUnless(has_einops, "Requires einops")
    def test_load_old_state_dict_proj_attn_discarded_when_no_out_proj(self):
        params = {**self._MIGRATION_PARAMS, "include_fc": False}
        src = AutoencoderKL(**params).to(device)
        old_sd = self._new_to_old_sd(src.state_dict(), include_proj_attn=False)

        # inject synthetic proj_attn keys (mimic an old checkpoint)
        attn_blocks = [k.replace(".to_q.weight", "") for k in old_sd if k.endswith(".to_q.weight")]
        self.assertGreater(len(attn_blocks), 0, "No attention blocks found - check model config")
        for block in attn_blocks:
            ch = old_sd[f"{block}.to_q.weight"].shape[0]
            old_sd[f"{block}.proj_attn.weight"] = torch.randn(ch, ch)
            old_sd[f"{block}.proj_attn.bias"] = torch.randn(ch)

        dst = AutoencoderKL(**params).to(device)
        dst.load_old_state_dict(old_sd)

        loaded = dst.state_dict()
        self.assertFalse(
            any("out_proj" in k for k in loaded), "out_proj should not exist in a model built with include_fc=False"
        )


if __name__ == "__main__":
    unittest.main()
