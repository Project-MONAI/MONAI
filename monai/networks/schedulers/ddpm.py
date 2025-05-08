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

import numpy as np
import torch

from monai.utils import StrEnum

from .scheduler import Scheduler


class DDPMVarianceType(StrEnum):
    """
    Valid names for DDPM Scheduler's `variance_type` argument. Options to clip the variance used when adding noise
    to the denoised sample.
    """

    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED = "learned"
    LEARNED_RANGE = "learned_range"


class DDPMPredictionType(StrEnum):
    """
    Set of valid prediction type names for the DDPM scheduler's `prediction_type` argument.

    epsilon: predicting the noise of the diffusion process
    sample: directly predicting the noisy sample
    v_prediction: velocity prediction, see section 2.4 https://imagen.research.google/video/paper.pdf
    """

    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"


class DDPMScheduler(Scheduler):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling. Based on: Ho et al., "Denoising Diffusion Probabilistic Models"
    https://arxiv.org/abs/2006.11239

    Args:
        num_train_timesteps: number of diffusion steps used to train the model.
        schedule: member of NoiseSchedules, name of noise schedule function in component store
        variance_type: member of DDPMVarianceType
        clip_sample: option to clip predicted sample between -1 and 1 for numerical stability.
        prediction_type: member of DDPMPredictionType
        clip_sample_min: minimum clipping value when clip_sample equals True
        clip_sample_max: maximum clipping value when clip_sample equals True
        schedule_args: arguments to pass to the schedule function
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        schedule: str = "linear_beta",
        variance_type: str = DDPMVarianceType.FIXED_SMALL,
        clip_sample: bool = True,
        prediction_type: str = DDPMPredictionType.EPSILON,
        clip_sample_min: float = -1.0,
        clip_sample_max: float = 1.0,
        **schedule_args,
    ) -> None:
        super().__init__(num_train_timesteps, schedule, **schedule_args)

        if variance_type not in DDPMVarianceType.__members__.values():
            raise ValueError("Argument `variance_type` must be a member of `DDPMVarianceType`")

        if prediction_type not in DDPMPredictionType.__members__.values():
            raise ValueError("Argument `prediction_type` must be a member of `DDPMPredictionType`")

        self.clip_sample = clip_sample
        self.clip_sample_values = [clip_sample_min, clip_sample_max]
        self.variance_type = variance_type
        self.prediction_type = prediction_type

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps: number of diffusion steps used when generating samples with a pre-trained model.
            device: target device to put the data.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_mean(self, timestep: int, x_0: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of the posterior at timestep t.

        Args:
            timestep: current timestep.
            x0: the noise-free input.
            x_t: the input noised to timestep t.

        Returns:
            Returns the mean
        """
        # these attributes are used for calculating the posterior, q(x_{t-1}|x_t,x_0),
        # (see formula (5-7) from https://arxiv.org/pdf/2006.11239.pdf)
        alpha_t = self.alphas[timestep]
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one

        x_0_coefficient = alpha_prod_t_prev.sqrt() * self.betas[timestep] / (1 - alpha_prod_t)
        x_t_coefficient = alpha_t.sqrt() * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)

        mean: torch.Tensor = x_0_coefficient * x_0 + x_t_coefficient * x_t

        return mean

    def _get_variance(self, timestep: int, predicted_variance: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute the variance of the posterior at timestep t.

        Args:
            timestep: current timestep.
            predicted_variance: variance predicted by the model.

        Returns:
            Returns the variance
        """
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance: torch.Tensor = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[timestep]
        # hacks - were probably added for training stability
        if self.variance_type == DDPMVarianceType.FIXED_SMALL:
            variance = torch.clamp(variance, min=1e-20)
        elif self.variance_type == DDPMVarianceType.FIXED_LARGE:
            variance = self.betas[timestep]
        elif self.variance_type == DDPMVarianceType.LEARNED and predicted_variance is not None:
            return predicted_variance
        elif self.variance_type == DDPMVarianceType.LEARNED_RANGE and predicted_variance is not None:
            min_log = variance
            max_log = self.betas[timestep]
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            generator: random number generator.

        Returns:
            pred_prev_sample: Predicted previous sample
        """
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == DDPMPredictionType.EPSILON:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == DDPMPredictionType.SAMPLE:
            pred_original_sample = model_output
        elif self.prediction_type == DDPMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, self.clip_sample_values[0], self.clip_sample_values[1]
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance: torch.Tensor = torch.tensor(0)
        if timestep > 0:
            noise = torch.randn(
                model_output.size(),
                dtype=model_output.dtype,
                layout=model_output.layout,
                generator=generator,
                device=model_output.device,
            )
            variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample
