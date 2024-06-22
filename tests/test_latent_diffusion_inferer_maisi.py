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

import torch
from parameterized import parameterized

from monai.inferers import LatentDiffusionInfererMaisi
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.utils import optional_import

_, has_einops = optional_import("einops")
TEST_CASES = [
    [
        "AutoencoderKL",
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "channels": (4, 4),
            "latent_channels": 3,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "norm_num_groups": 4,
        },
        "DiffusionModelUNet",
        {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": [4, 4],
            "norm_num_groups": 4,
            "attention_levels": [False, False],
            "num_res_blocks": 1,
            "num_head_channels": 4,
        },
        (1, 1, 8, 8),
        (1, 3, 4, 4),
    ],
]


class TestLatentDiffusionInfererMaisi(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()

        prediction = inferer(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            noise=noise,
            timesteps=timesteps,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_shape(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        noise = torch.randn(latent_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample = inferer.sample(
            input_noise=noise,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_intermediates(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        noise = torch.randn(latent_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.sample(
            input_noise=noise,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, input_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_get_likelihoods(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.get_likelihood(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_resample_likelihoods(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample, intermediates = inferer.get_likelihood(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            save_intermediates=True,
            resample_latent_likelihoods=True,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)
        self.assertEqual(intermediates[0].shape[2:], input_shape[2:])

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_prediction_shape_conditioned_concat(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2_params = stage_2_params.copy()
        n_concat_channel = 3
        stage_2_params["in_channels"] = stage_2_params["in_channels"] + n_concat_channel
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        input = torch.randn(input_shape).to(device)
        noise = torch.randn(latent_shape).to(device)
        conditioning_shape = list(latent_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)

        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()

        prediction = inferer(
            inputs=input,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            noise=noise,
            timesteps=timesteps,
            condition=conditioning,
            mode="concat",
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(prediction.shape, latent_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_shape_conditioned_concat(
        self, ae_model_type, autoencoder_params, dm_model_type, stage_2_params, input_shape, latent_shape
    ):
        if ae_model_type == "AutoencoderKL":
            stage_1 = AutoencoderKL(**autoencoder_params)
        else:
            raise ValueError(f"Unsupported autoencoder model type: {ae_model_type}")
        
        stage_2_params = stage_2_params.copy()
        n_concat_channel = 3
        stage_2_params["in_channels"] = stage_2_params["in_channels"] + n_concat_channel
        stage_2 = DiffusionModelUNet(**stage_2_params) if dm_model_type == "DiffusionModelUNet" else None
        if stage_2 is None:
            raise ValueError(f"Unsupported diffusion model type: {dm_model_type}")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        stage_1.to(device)
        stage_2.to(device)
        stage_1.eval()
        stage_2.eval()

        noise = torch.randn(latent_shape).to(device)
        conditioning_shape = list(latent_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)
        top_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        bottom_region_index_tensor = torch.randint(0, 2, input_shape).to(device)
        spacing_tensor = torch.randn(input_shape).to(device)

        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = LatentDiffusionInfererMaisi(scheduler=scheduler, scale_factor=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample = inferer.sample(
            input_noise=noise,
            autoencoder_model=stage_1,
            diffusion_model=stage_2,
            scheduler=scheduler,
            conditioning=conditioning,
            mode="concat",
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(sample.shape, input_shape)


if __name__ == "__main__":
    unittest.main()
