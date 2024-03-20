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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.cuda.amp import autocast

from monai.networks import eval_mode
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet
from monai.networks.schedulers import DDIMScheduler
from monai.utils import optional_import
from tests.utils import SkipIfBeforePyTorchVersion

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CASES = [
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
            "attention_levels": (False, False, True),
            "num_res_blocks": 1,
            "norm_num_groups": 4,
        },
        (1, 1, 16, 16, 16),
        (1, 1, 16, 16, 16),
        (1, 4, 4, 4, 4),
    ],
]


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

    def test_load_old_weights(self):
        new_attention = True
        net = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(128, 256, 384),
            latent_channels=8,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
            new_attention=new_attention,
        ).to(device)

        old_state_dict = torch.load("/home/mark/data_drive/monai/autoencoderkl/autoencoder_kl_4.pth")
        if new_attention:
            new_state_dict = net.state_dict()
            for k in new_state_dict:
                if k in old_state_dict:
                    new_state_dict[k] = old_state_dict[k]
                # else:
                #    print(f"key {k} not found in old state dict")
            # get all prefixes for attention blocks
            attention_blocks = [k.replace(".attn.qkv.weight", "") for k in new_state_dict if "attn.qkv.weight" in k]
            for block in attention_blocks:
                new_state_dict[f"{block}.attn.qkv.weight"] = torch.concat(
                    [
                        old_state_dict[f"{block}.to_q.weight"],
                        old_state_dict[f"{block}.to_k.weight"],
                        old_state_dict[f"{block}.to_v.weight"],
                    ],
                    dim=0,
                )
                new_state_dict[f"{block}.attn.qkv.bias"] = torch.concat(
                    [
                        old_state_dict[f"{block}.to_q.bias"],
                        old_state_dict[f"{block}.to_k.bias"],
                        old_state_dict[f"{block}.to_v.bias"],
                    ],
                    dim=0,
                )
                # looks like the old weights were not used
                new_state_dict[f"{block}.attn.out_proj.weight"] = torch.eye(
                    new_state_dict[f"{block}.attn.out_proj.weight"].shape[0]
                )  # old_state_dict[f"{block}.proj_attn.weight"]
                new_state_dict[f"{block}.attn.out_proj.bias"] = torch.zeros(
                    new_state_dict[f"{block}.attn.out_proj.bias"].shape
                )  # old_state_dict[f"{block}.proj_attn.bias"]
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(old_state_dict)
        with eval_mode(net):
            # set torch seed for reproducibility
            expected = torch.tensor(
                [
                    [
                        [
                            [0.2762, 0.3151, 0.4972, 0.6143, 0.7014, 0.7171, 0.6284, 0.3858],
                            [0.1632, 0.1336, 0.2328, 0.2730, 0.3836, 0.5495, 0.6277, 0.6333],
                            [0.2129, 0.3053, 0.3755, 0.4203, 0.4529, 0.6616, 0.7911, 0.8150],
                            [0.2801, 0.3676, 0.4372, 0.4962, 0.4788, 0.6480, 0.8440, 0.8402],
                            [0.2997, 0.3663, 0.3973, 0.4385, 0.4347, 0.5638, 0.7304, 0.7145],
                            [0.3708, 0.4936, 0.5168, 0.5270, 0.5152, 0.5899, 0.6683, 0.6483],
                            [0.3498, 0.5183, 0.5831, 0.6051, 0.6186, 0.6297, 0.5913, 0.5578],
                            [0.2338, 0.3903, 0.4052, 0.4612, 0.5245, 0.5308, 0.5028, 0.4065],
                        ]
                    ]
                ],
                device="cuda:0",
            )
            torch.manual_seed(0)
            input = torch.randn((1, 1, 8, 8)).to(device)
            result, _, _ = net.forward(input)
            self.assertEqual(result.shape, expected.shape)
            self.assertTrue(torch.allclose(result, expected, atol=1e-4))
            print("debug")

    def test_load_brain_image_synthesis_ldm(self):
        new_attention = True
        net = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(64, 128, 128, 128),
            latent_channels=3,
            num_res_blocks=2,
            norm_num_groups=32,
            attention_levels=(False, False, False, False),
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        ).to(device)

        old_state_dict = torch.load(
            "/home/mark/projects/monai/models/brain_image_synthesis_latent_diffusion_model/autoencoder.pth"
        )
        if new_attention:
            new_state_dict = net.state_dict()
            for k in new_state_dict:
                if k in old_state_dict:
                    new_state_dict[k] = old_state_dict[k]
                # else:
                #    print(f"key {k} not found in old state dict")
            # get all prefixes for attention blocks
            attention_blocks = [k.replace(".attn.qkv.weight", "") for k in new_state_dict if "attn.qkv.weight" in k]
            for block in attention_blocks:
                new_state_dict[f"{block}.attn.qkv.weight"] = torch.concat(
                    [
                        old_state_dict[f"{block}.to_q.weight"],
                        old_state_dict[f"{block}.to_k.weight"],
                        old_state_dict[f"{block}.to_v.weight"],
                    ],
                    dim=0,
                )
                new_state_dict[f"{block}.attn.qkv.bias"] = torch.concat(
                    [
                        old_state_dict[f"{block}.to_q.bias"],
                        old_state_dict[f"{block}.to_k.bias"],
                        old_state_dict[f"{block}.to_v.bias"],
                    ],
                    dim=0,
                )
                # looks like the old weights were not used
                new_state_dict[f"{block}.attn.out_proj.weight"] = torch.eye(
                    new_state_dict[f"{block}.attn.out_proj.weight"].shape[0]
                )  # old_state_dict[f"{block}.proj_attn.weight"]
                new_state_dict[f"{block}.attn.out_proj.bias"] = torch.zeros(
                    new_state_dict[f"{block}.attn.out_proj.bias"].shape
                )  # old_state_dict[f"{block}.proj_attn.bias"]
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(old_state_dict)

        # get the diffusion model with these params
        diffusion_model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=7,
            out_channels=3,
            channels=[256, 512, 768],
            num_res_blocks=2,
            attention_levels=[False, True, True],
            norm_num_groups=32,
            norm_eps=1e-06,
            resblock_updown=True,
            num_head_channels=[0, 512, 768],
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=4,
            upcast_attention=True,
            use_flash_attention=False,
        ).to(device)
        diffusion_model.load_state_dict(
            torch.load(
                "/home/mark/projects/monai/models/brain_image_synthesis_latent_diffusion_model/diffusion_model.pth"
            )
        )

        # do sampling
        # gender, age, ventricular model, brain vol
        conditioning = torch.tensor([[0.0, 0.1, 1.0, 0.4]]).to(device).unsqueeze(1)
        scheduler = DDIMScheduler(
            beta_start=0.0015,
            beta_end=0.0205,
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            clip_sample=False,
        )
        scheduler.set_timesteps(50)
        noise = torch.randn((1, 3, 20, 28, 20)).to(device)
        sample = sampling_fn(noise, net, diffusion_model, scheduler, conditioning)

        # plot several slices through the volume
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(sample[0, 0, :, :, 80].cpu().detach().numpy(), cmap="gray")
        axs[1].imshow(sample[0, 0, :, 110, :].cpu().detach().numpy(), cmap="gray")
        axs[2].imshow(sample[0, 0, 80, :, :].cpu().detach().numpy(), cmap="gray")
        plt.show()
        print("debug")


@torch.no_grad()
def sampling_fn(
    input_noise: torch.Tensor,
    autoencoder_model: nn.Module,
    diffusion_model: nn.Module,
    scheduler: nn.Module,
    conditioning: torch.Tensor,
) -> torch.Tensor:
    if has_tqdm:
        progress_bar = tqdm(scheduler.timesteps)
    else:
        progress_bar = iter(scheduler.timesteps)

    image = input_noise
    cond_concat = conditioning.squeeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    cond_concat = cond_concat.expand(list(cond_concat.shape[0:2]) + list(input_noise.shape[2:]))
    for t in progress_bar:
        with torch.no_grad():
            model_output = diffusion_model(
                torch.cat((image, cond_concat), dim=1),
                timesteps=torch.Tensor((t,)).to(input_noise.device).long(),
                context=conditioning,
            )
            image, _ = scheduler.step(model_output, t, image)

    with torch.no_grad():
        with autocast():
            sample = autoencoder_model.decode_stage_2_outputs(image)

    return sample


if __name__ == "__main__":
    unittest.main()
