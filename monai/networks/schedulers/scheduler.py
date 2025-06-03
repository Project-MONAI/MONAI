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
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
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

import torch
import torch.nn as nn

from monai.utils import ComponentStore, unsqueeze_right

NoiseSchedules = ComponentStore("NoiseSchedules", "Functions to generate noise schedules")


@NoiseSchedules.add_def("linear_beta", "Linear beta schedule")
def _linear_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """
    Linear beta noise schedule function.

    Args:
        num_train_timesteps: number of timesteps
        beta_start: start of beta range, default 1e-4
        beta_end: end of beta range, default 2e-2

    Returns:
        betas: beta schedule tensor
    """
    return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)


@NoiseSchedules.add_def("scaled_linear_beta", "Scaled linear beta schedule")
def _scaled_linear_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """
    Scaled linear beta noise schedule function.

    Args:
        num_train_timesteps: number of timesteps
        beta_start: start of beta range, default 1e-4
        beta_end: end of beta range, default 2e-2

    Returns:
        betas: beta schedule tensor
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2


@NoiseSchedules.add_def("sigmoid_beta", "Sigmoid beta schedule")
def _sigmoid_beta(num_train_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2, sig_range: float = 6):
    """
    Sigmoid beta noise schedule function.

    Args:
        num_train_timesteps: number of timesteps
        beta_start: start of beta range, default 1e-4
        beta_end: end of beta range, default 2e-2
        sig_range: pos/neg range of sigmoid input, default 6

    Returns:
        betas: beta schedule tensor
    """
    betas = torch.linspace(-sig_range, sig_range, num_train_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


@NoiseSchedules.add_def("cosine", "Cosine schedule")
def _cosine_beta(num_train_timesteps: int, s: float = 8e-3):
    """
    Cosine noise schedule, see https://arxiv.org/abs/2102.09672

    Args:
        num_train_timesteps: number of timesteps
        s: smoothing factor, default 8e-3 (see referenced paper)

    Returns:
        (betas, alphas, alpha_cumprod) values
    """
    x = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
    alphas_cumprod = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod /= alphas_cumprod[0].item()
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0, 0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


class Scheduler(nn.Module):
    """
    Base class for other schedulers based on a noise schedule function.

    This class is meant as the base for other schedulers which implement their own way of sampling or stepping. Here
    the class defines beta, alpha, and alpha_cumprod values from a noise schedule function named with `schedule`,
    which is the name of a component in NoiseSchedules. These components must all be callables which return either
    the beta schedule alone or a triple containing (betas, alphas, alphas_cumprod) values. New schedule functions
    can be provided by using the NoiseSchedules.add_def, for example:

    .. code-block:: python

        from monai.networks.schedulers import NoiseSchedules, DDPMScheduler

        @NoiseSchedules.add_def("my_beta_schedule", "Some description of your function")
        def _beta_function(num_train_timesteps, beta_start=1e-4, beta_end=2e-2):
            return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="my_beta_schedule")

    All such functions should have an initial positional integer argument `num_train_timesteps` stating the number of
    timesteps the schedule is for, otherwise any other arguments can be given which will be passed by keyword through
    the constructor's `schedule_args` value. To see what noise functions are available, print the object NoiseSchedules
    to get a listing of stored objects with their docstring descriptions.

    Note: in previous versions of the schedulers the argument `schedule_beta` was used to state the beta schedule
    type, this now replaced with `schedule` and most names used with the previous argument now have "_beta" appended
    to them, eg. 'schedule_beta="linear"' -> 'schedule="linear_beta"'. The `beta_start` and `beta_end` arguments are
    still used for some schedules but these are provided as keyword arguments now.

    Args:
        num_train_timesteps: number of diffusion steps used to train the model.
        schedule: member of NoiseSchedules,
            a named function returning the beta tensor or (betas, alphas, alphas_cumprod) triple
        schedule_args: arguments to pass to the schedule function
    """

    def __init__(self, num_train_timesteps: int = 1000, schedule: str = "linear_beta", **schedule_args) -> None:
        super().__init__()
        schedule_args["num_train_timesteps"] = num_train_timesteps
        noise_sched = NoiseSchedules[schedule](**schedule_args)

        # set betas, alphas, alphas_cumprod based off return value from noise function
        if isinstance(noise_sched, tuple):
            self.betas, self.alphas, self.alphas_cumprod = noise_sched
        else:
            self.betas = noise_sched
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.num_train_timesteps = num_train_timesteps
        self.one = torch.tensor(1.0)

        # settable values
        self.num_inference_steps: int | None = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Add noise to the original samples.

        Args:
            original_samples: original samples
            noise: noise to add to samples
            timesteps: timesteps tensor indicating the timestep to be computed for each sample.

        Returns:
            noisy_samples: sample with added noise
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_cumprod: torch.Tensor = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, original_samples.ndim)
        sqrt_one_minus_alpha_prod: torch.Tensor = unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, original_samples.ndim
        )

        noisy_samples = sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod: torch.Tensor = unsqueeze_right(self.alphas_cumprod[timesteps] ** 0.5, sample.ndim)
        sqrt_one_minus_alpha_prod: torch.Tensor = unsqueeze_right(
            (1 - self.alphas_cumprod[timesteps]) ** 0.5, sample.ndim
        )

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
