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

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

UNCOND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": (1, 1, 2),
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, True, True),
            "num_head_channels": (0, 2, 4),
            "norm_num_groups": 8,
        }
    ],
]

UNCOND_CASES_3D = [
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, False),
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
        }
    ],
    [
        {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": (0, 0, 4),
            "norm_num_groups": 8,
        }
    ],
]

COND_CASES_2D = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "resblock_updown": True,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        }
    ],
]

DROPOUT_OK = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "dropout_cattn": 0.25,
        }
    ],
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
        }
    ],
]

DROPOUT_WRONG = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": 1,
            "channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 4,
            "norm_num_groups": 8,
            "with_conditioning": True,
            "transformer_num_layers": 1,
            "cross_attention_dim": 3,
            "dropout_cattn": 3.0,
        }
    ]
]


class TestDiffusionModelUNet2D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_2D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16))

    def test_timestep_with_wrong_shape(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1, 1)).long())

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, False),
            norm_num_groups=8,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 16, 16))

    def test_model_channels_not_multiple_of_norm_num_group(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 12),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    def test_attention_levels_with_different_length_num_head_channels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False, False),
                num_head_channels=(0, 2),
                norm_num_groups=8,
            )

    def test_num_res_blocks_with_different_length_channels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(1, 1),
                channels=(8, 8, 8),
                attention_levels=(False, False, False),
                norm_num_groups=8,
            )

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
            norm_num_groups=8,
            num_head_channels=8,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    def test_with_conditioning_cross_attention_dim_none(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False, True),
                with_conditioning=True,
                transformer_num_layers=1,
                cross_attention_dim=None,
                norm_num_groups=8,
            )

    def test_context_with_conditioning_none(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            with_conditioning=False,
            transformer_num_layers=1,
            norm_num_groups=8,
        )

        with self.assertRaises(ValueError):
            with eval_mode(net):
                net.forward(
                    x=torch.rand((1, 1, 16, 32)),
                    timesteps=torch.randint(0, 1000, (1,)).long(),
                    context=torch.rand((1, 1, 3)),
                )

    def test_shape_conditioned_models_class_conditioning(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 32)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                class_labels=torch.randint(0, 2, (1,)).long(),
            )
            self.assertEqual(result.shape, (1, 1, 16, 32))

    def test_conditioned_models_no_class_labels(self):
        net = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=8,
            num_head_channels=8,
            num_class_embeds=2,
        )

        with self.assertRaises(ValueError):
            net.forward(x=torch.rand((1, 1, 16, 32)), timesteps=torch.randint(0, 1000, (1,)).long())

    def test_model_channels_not_same_size_of_attention_levels(self):
        with self.assertRaises(ValueError):
            DiffusionModelUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=1,
                channels=(8, 8, 8),
                attention_levels=(False, False),
                norm_num_groups=8,
                num_head_channels=8,
                num_class_embeds=2,
            )

    @parameterized.expand(COND_CASES_2D)
    def test_conditioned_2d_models_shape(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 1, 3)))
            self.assertEqual(result.shape, (1, 1, 16, 16))


class TestDiffusionModelUNet3D(unittest.TestCase):
    @parameterized.expand(UNCOND_CASES_3D)
    def test_shape_unconditioned_models(self, input_param):
        net = DiffusionModelUNet(**input_param)
        with eval_mode(net):
            result = net.forward(torch.rand((1, 1, 16, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    def test_shape_with_different_in_channel_out_channel(self):
        in_channels = 6
        out_channels = 3
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=1,
            channels=(8, 8, 8),
            attention_levels=(False, False, True),
            norm_num_groups=4,
        )
        with eval_mode(net):
            result = net.forward(torch.rand((1, in_channels, 16, 16, 16)), torch.randint(0, 1000, (1,)).long())
            self.assertEqual(result.shape, (1, out_channels, 16, 16, 16))

    def test_shape_conditioned_models(self):
        net = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=1,
            channels=(16, 16, 16),
            attention_levels=(False, False, True),
            norm_num_groups=16,
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=3,
        )
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 1, 16, 16, 16)),
                timesteps=torch.randint(0, 1000, (1,)).long(),
                context=torch.rand((1, 1, 3)),
            )
            self.assertEqual(result.shape, (1, 1, 16, 16, 16))

    # Test dropout specification for cross-attention blocks
    @parameterized.expand(DROPOUT_WRONG)
    def test_wrong_dropout(self, input_param):
        with self.assertRaises(ValueError):
            _ = DiffusionModelUNet(**input_param)

    @parameterized.expand(DROPOUT_OK)
    def test_right_dropout(self, input_param):
        _ = DiffusionModelUNet(**input_param)

    def test_compatability(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = DiffusionModelUNet(
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
        ).to(device)
        net.load_old_state_dict(
            torch.load(
                "/home/mark/projects/monai/models/brain_image_synthesis_latent_diffusion_model/diffusion_model.pth"
            ),
            verbose=True,
        )
        # set random seed
        torch.manual_seed(0)
        expected = torch.Tensor(
            [
                [
                    [[0.2826, -0.2253], [0.4766, 0.6938], [-0.0498, 0.5390]],
                    [[0.1214, 0.6526], [0.6149, 0.5114], [0.6663, 0.1482]],
                ],
                [
                    [[0.2761, -0.1863], [0.0352, 0.0768], [0.5582, -0.0790]],
                    [[0.0351, 0.2923], [0.2157, 0.2395], [0.1058, 0.6407]],
                ],
            ]
        ).to(device)
        with eval_mode(net):
            result = net.forward(
                x=torch.rand((1, 7, 20, 28, 20)).to(device),
                timesteps=torch.randint(0, 1000, (1,)).long().to(device),
                context=torch.rand((1, 1, 4)).to(device),
            )
            assert torch.allclose(result[0, ::2, ::12, ::12, ::12], expected, atol=1e-3)

    def test_load_brain_image_synthesis_ldm(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        net.load_old_state_dict(old_state_dict)

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
        ).to(device)
        diffusion_model.load_old_state_dict(
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
