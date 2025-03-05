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
#
# =========================================================================
# Adapted from https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/rf/rectified_flow.py
# which has the following license:
# https://github.com/hpcaitech/Open-Sora/blob/main/LICENSE
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.distributions import LogisticNormal

from .scheduler import Scheduler


def timestep_transform(
    t, input_img_size_numel, base_img_size_numel=32 * 32 * 32, scale=1.0, num_train_timesteps=1000, spatial_dim=3
):
    """
    Applies a transformation to the timestep based on image resolution scaling.

    Args:
        t (torch.Tensor): The original timestep(s).
        input_img_size_numel (torch.Tensor): The input image's size (H * W * D).
        base_img_size_numel (int): reference H*W*D size, usually smaller than input_img_size_numel.
        scale (float): Scaling factor for the transformation.
        num_train_timesteps (int): Total number of training timesteps.
        spatial_dim (int): Number of spatial dimensions in the image.

    Returns:
        torch.Tensor: Transformed timestep(s).
    """
    t = t / num_train_timesteps
    ratio_space = (input_img_size_numel / base_img_size_numel).pow(1.0 / spatial_dim)

    ratio = ratio_space * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_train_timesteps
    return new_t


class RFlowScheduler(Scheduler):
    """
    A rectified flow scheduler for guiding the diffusion process in a generative model.

    Supports uniform and logit-normal sampling methods, timestep transformation for
    different resolutions, and noise addition during diffusion.

    Attributes:
        num_train_timesteps (int): Total number of training timesteps.
        use_discrete_timesteps (bool): Whether to use discrete timesteps.
        sample_method (str): Training time step sampling method ('uniform' or 'logit-normal').
        loc (float): Location parameter for logit-normal distribution, used only if sample_method='logit-normal'.
        scale (float): Scale parameter for logit-normal distribution, used only if sample_method='logit-normal'.
        use_timestep_transform (bool): Whether to apply timestep transformation.
            If true, there will be more inference timesteps at early(noisy) stages for larger image volumes.
        transform_scale (float): Scaling factor for timestep transformation, used only if use_timestep_transform=True.
        steps_offset (int): Offset added to computed timesteps, used only if use_timestep_transform=True.
        base_img_size_numel (int): Reference image volume size for scaling, used only if use_timestep_transform=True.

    Example:

        .. code-block:: python

            # define a scheduler
            noise_scheduler = RFlowScheduler(
                num_train_timesteps = 1000,
                use_discrete_timesteps = True,
                sample_method = 'logit-normal',
                use_timestep_transform = True,
                base_img_size_numel = 32 * 32 * 32
            )

            # during training
            inputs = torch.ones(2,4,64,64,32)
            noise = torch.randn_like(inputs)
            timesteps = noise_scheduler.sample_timesteps(inputs)
            noisy_inputs = noise_scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            predicted_velocity = diffusion_unet(
                x=noisy_inputs,
                timesteps=timesteps
            )
            loss = loss_l1(predicted_velocity, (inputs - noise))

            # during inference
            noisy_inputs = torch.randn(2,4,64,64,32)
            input_img_size_numel = torch.prod(torch.tensor(noisy_inputs.shape[-3:])
            noise_scheduler.set_timesteps(
                num_inference_steps=30, input_img_size_numel=input_img_size_numel)
            )
            all_next_timesteps = torch.cat(
                (noise_scheduler.timesteps[1:], torch.tensor([0], dtype=noise_scheduler.timesteps.dtype))
            )
            for t, next_t in tqdm(
                zip(noise_scheduler.timesteps, all_next_timesteps),
                total=min(len(noise_scheduler.timesteps), len(all_next_timesteps)),
            ):
                predicted_velocity = diffusion_unet(
                    x=noisy_inputs,
                    timesteps=timesteps
                )
                noisy_inputs, _ = noise_scheduler.step(predicted_velocity, t, noisy_inputs, next_t)
            final_output = noisy_inputs
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        use_discrete_timesteps: bool = True,
        sample_method: str = "uniform",
        loc: float = 0.0,
        scale: float = 1.0,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        steps_offset: int = 0,
        base_img_size_numel: int = 32 * 32 * 32,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.use_discrete_timesteps = use_discrete_timesteps
        self.base_img_size_numel = base_img_size_numel

        # sample method
        if sample_method not in ["uniform", "logit-normal"]:
            raise ValueError(
                f"sample_method = {sample_method}, which has to be chosen from ['uniform', 'logit-normal']."
            )
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        self.steps_offset = steps_offset

    def add_noise(
        self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        Adds noise to the original samples based on the given timesteps.

        Args:
            original_samples (torch.FloatTensor): The original sample tensor.
            noise (torch.FloatTensor): Noise tensor to be added.
            timesteps (torch.IntTensor): Timesteps corresponding to each sample.

        Returns:
            torch.FloatTensor: The noisy sample tensor.
        """
        timepoints = timesteps.float() / self.num_train_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        input_img_size_numel: int | None = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
            device: target device to put the data.
            input_img_size_numel: int, H*W*D of the image, used with self.use_timestep_transform is True.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        # prepare timesteps
        timesteps = [
            (1.0 - i / self.num_inference_steps) * self.num_train_timesteps for i in range(self.num_inference_steps)
        ]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    input_img_size_numel=input_img_size_numel,
                    base_img_size_numel=self.base_img_size_numel,
                    num_train_timesteps=self.num_train_timesteps,
                )
                for t in timesteps
            ]
        timesteps = np.array(timesteps).astype(np.float16)
        if self.use_discrete_timesteps:
            timesteps = timesteps.astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset

    def sample_timesteps(self, x_start):
        """
        Randomly samples training timesteps using the chosen sampling method.

        Args:
            x_start (torch.Tensor): The input tensor for sampling.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        if self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_train_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * self.num_train_timesteps

        if self.use_discrete_timesteps:
            t = t.long()

        if self.use_timestep_transform:
            input_img_size_numel = torch.prod(torch.tensor(x_start.shape[-3:]))
            t = timestep_transform(
                t,
                input_img_size_numel=input_img_size_numel,
                base_img_size_numel=self.base_img_size_numel,
                num_train_timesteps=self.num_train_timesteps,
            )

        return t

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, next_timestep=None
    ) -> tuple[torch.Tensor, Any]:
        """
        Predict the sample at the previous timestep. Core function to propagate the diffusion
        process from the learned model outputs.

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            next_timestep: next discrete timestep in the diffusion chain.
        Returns:
            pred_prev_sample: Predicted previous sample
            None
        """
        v_pred = model_output
        if next_timestep is None:
            dt = 1.0 / self.num_inference_steps
        else:
            dt = timestep - next_timestep
            dt = dt / self.num_train_timesteps
        z = sample + v_pred * dt

        return z, None
