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
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler
from parameterized import parameterized

from monai.apps.generation.maisi.inferers.inferer_maisi import DiffusionInfererMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.utils import optional_import

_, has_scipy = optional_import("scipy")
_, has_einops = optional_import("einops")

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8),
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": [8],
            "norm_num_groups": 8,
            "attention_levels": [True],
            "num_res_blocks": 1,
            "num_head_channels": 8,
        },
        (2, 1, 8, 8, 8),
    ],
]


class TestDiffusionInfererMaisi(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_call(self, model_params, input_shape):
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        input = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample = inferer(
            inputs=input,
            noise=noise,
            diffusion_model=model,
            timesteps=timesteps,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(sample.shape, input_shape)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sample_intermediates(self, model_params, input_shape):
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_ddpm_sampler(self, model_params, input_shape):
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_ddim_sampler(self, model_params, input_shape):
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sampler_conditioned(self, model_params, input_shape):
        model_params["with_conditioning"] = True
        model_params["cross_attention_dim"] = 3
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        conditioning = torch.randn([input_shape[0], 1, 3]).to(device)
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            conditioning=conditioning,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_sampler_conditioned_concat(self, model_params, input_shape):
        model_params = model_params.copy()
        n_concat_channel = 2
        model_params["in_channels"] = model_params["in_channels"] + n_concat_channel
        model_params["cross_attention_dim"] = None
        model_params["with_conditioning"] = False
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        noise = torch.randn(input_shape).to(device)
        conditioning_shape = list(input_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=model,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=1,
            conditioning=conditioning,
            mode="concat",
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(len(intermediates), 10)

    @parameterized.expand(TEST_CASES)
    @skipUnless(has_einops, "Requires einops")
    def test_call_conditioned_concat(self, model_params, input_shape):
        model_params = model_params.copy()
        n_concat_channel = 2
        model_params["in_channels"] = model_params["in_channels"] + n_concat_channel
        model_params["cross_attention_dim"] = None
        model_params["with_conditioning"] = False
        model = DiffusionModelUNetMaisi(**model_params)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        input = torch.randn(input_shape).to(device)
        noise = torch.randn(input_shape).to(device)
        conditioning_shape = list(input_shape)
        conditioning_shape[1] = n_concat_channel
        conditioning = torch.randn(conditioning_shape).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=10)
        inferer = DiffusionInfererMaisi(scheduler=scheduler)
        scheduler.set_timesteps(num_inference_steps=10)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (input_shape[0],), device=input.device).long()
        top_region_index_tensor = torch.rand((1, 4)).to(device)
        bottom_region_index_tensor = torch.rand((1, 4)).to(device)
        spacing_tensor = torch.rand((1, 3)).to(device)
        sample = inferer(
            inputs=input,
            noise=noise,
            diffusion_model=model,
            timesteps=timesteps,
            condition=conditioning,
            mode="concat",
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )
        self.assertEqual(sample.shape, input_shape)


if __name__ == "__main__":
    unittest.main()
