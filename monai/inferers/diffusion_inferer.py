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

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from pydoc import locate
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.apps.utils import get_logger
from monai.data import decollate_batch
from monai.inferers.inferer import Inferer
from monai.inferers.utils import compute_importance_map, sliding_window_inference
from monai.networks.nets import (
    VQVAE,
    AutoencoderKL,
    ControlNet,
    DiffusionModelUNet,
    SPADEAutoencoderKL,
    SPADEDiffusionModelUNet,
)
from monai.networks.schedulers import RFlowScheduler, Scheduler
from monai.transforms import CenterSpatialCrop, SpatialPad
from monai.utils import ensure_tuple, optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

logger = get_logger(__name__)

__all__ = [
    "BaseDiffusionInferer",
    "DiffusionInferer",
    "LatentDiffusionInferer",
    "ControlNetDiffusionInferer",
    "ControlNetLatentDiffusionInferer",
]


class BaseDiffusionInferer(Inferer):
    """
    A base class for diffusion model inferers.

    """

    @abstractmethod
    def __init__(self, scheduler: Scheduler) -> None:
        """Initialise the diffusion model.

        Args:
            scheduler (Scheduler): scheduler used in combination with the diffusion model.
        """
        super().__init__()

    @abstractmethod
    def __call__(
        self,
        inputs: torch.Tensor,
        network: Callable,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        mode: str = "crossattn",
        condition: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:  # noqa: E501
        """
        Runs a forward pass on the diffusion model.

        Args:
            inputs: input to the model.
            network: diffusion model network.
            noise: noise to be added to the inputs.
            timesteps: timesteps to run inference for.
            mode: Conditioning mode for the network (concat or crossattn).
            condition: context tensor (if applicable).
            args: optional args to be passed to the inferer.
            kwargs: optional keyword args to be passed to the inferer.
        Raises:
            NotImplementedError: if the method is not implemented in this class.
        """

        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Runs inference on the diffusion model, by iterating across the diffusion chain.

        Args:
            input_noise (torch.Tensor): input noise to start inference.
            diffusion_model (DiffusionModelUNet): instance of diffusion model.
            scheduler (Scheduler | None, optional) diffusion scheduler. Defaults to None.
            save_intermediates (bool | None, optional): whether to output evenly-spaced intermediate samples. Defaults to False.
            intermediate_steps (int | None, optional): number of intermediate steps to obtain if save_intermediates
            is True. Defaults to 100.
            conditioning (torch.Tensor | None, optional): context tensor (if applicable). Defaults to None.
            mode (str, optional): Conditioning mode for the network (concat or crossattn). Defaults to "crossattn".
            verbose (bool, optional): whether inference process should be printed. Defaults to True.
            args: optional args to be passed to the inferer.
            kwargs: optional keyword args to be passed to the inferer.

        Raises:
            NotImplementedError: if the method is not implemented in this class
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


class DiffusionInferer(BaseDiffusionInferer):
    """Diffusion inference implementation for DiffusionModelUNet-based models."""

    def __init__(self, scheduler: Scheduler) -> None:  # type: ignore[override]
        super().__init__(scheduler)

        self.scheduler = scheduler

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        network: DiffusionModelUNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            network: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image: torch.Tensor = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        if mode == "concat":
            if condition is None:
                raise ValueError("Conditioning is required for concat condition")
            else:
                noisy_image = torch.cat([noisy_image, condition], dim=1)
                condition = None
        network = partial(network, seg=seg) if isinstance(network, SPADEDiffusionModelUNet) else network
        prediction: torch.Tensor = network(x=noisy_image, timesteps=timesteps, context=condition)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
            cfg: classifier-free-guidance scale, which indicates the level of strengthening on the conditioning.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if mode == "concat" and conditioning is None:
            raise ValueError("Conditioning must be supplied for if condition mode is concat.")
        if not scheduler:
            scheduler = self.scheduler
        image = input_noise

        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype)))
        if verbose and has_tqdm:
            progress_bar = tqdm(
                zip(scheduler.timesteps, all_next_timesteps),
                total=min(len(scheduler.timesteps), len(all_next_timesteps)),
            )
        else:
            progress_bar = iter(zip(scheduler.timesteps, all_next_timesteps))
        intermediates = []

        for t, next_t in progress_bar:
            # 1. predict noise model_output
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if (
                cfg is not None
            ):  # if classifier-free guidance is used, a conditioned and unconditioned bit is generated.
                model_input = torch.cat([image] * 2, dim=0)
                if conditioning is not None:
                    uncondition = torch.ones_like(conditioning)
                    uncondition.fill_(-1)
                    conditioning_input = torch.cat([uncondition, conditioning], dim=0)
                else:
                    conditioning_input = None
            else:
                model_input = image
                conditioning_input = conditioning
            if mode == "concat" and conditioning_input is not None:
                model_input = torch.cat([model_input, conditioning_input], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=None
                )
            else:
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning_input
                )
            if cfg is not None:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + cfg * (model_output_cond - model_output_uncond)

            # 2. compute previous image: x_t -> x_t-1
            if not isinstance(scheduler, RFlowScheduler):
                image, _ = scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = scheduler.step(model_output, t, image, next_t)  # type: ignore
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple = (0, 255),
        scaled_input_range: tuple = (0, 1),
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if mode == "concat" and conditioning is None:
            raise ValueError("Conditioning must be supplied for if condition mode is concat.")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat" and conditioning is not None:
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(noisy_image, timesteps=timesteps, context=None)
            else:
                model_output = diffusion_model(x=noisy_image, timesteps=timesteps, context=conditioning)

            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)  # type: ignore[operator]
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)  # type: ignore[operator]

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(dim=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0 + torch.tanh(torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device)) * (x + 0.044715 * torch.pow(x, 3)))
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple = (0, 255),
        scaled_input_range: tuple = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.

        Args:
            input: the target images. It is assumed that this was uint8 values,
                      rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        if inputs.shape != means.shape:
            raise ValueError(f"Inputs and means must have the same shape, got {inputs.shape} and {means.shape}")
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(inputs > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return log_probs


class LatentDiffusionInferer(DiffusionInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None, and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None and self.autoencoder_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=self.autoencoder_latent_shape)

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        network: DiffusionModelUNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            network: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        prediction: torch.Tensor = super().__call__(
            inputs=latent, network=network, noise=noise, timesteps=timesteps, condition=condition, mode=mode, seg=seg
        )
        return prediction

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        input_noise: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
            cfg: classifier-free-guidance scale, which indicates the level of strengthening on the conditioning.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                f"If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                f"labels for each must be compatible, but got {autoencoder_model.decoder.label_nc} and"
                f"{diffusion_model.label_nc}"
            )

        outputs = super().sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
            cfg=cfg,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            if save_intermediates:
                latent_intermediates = [
                    torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0)
                    for l in latent_intermediates
                ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
        image = decode(latent / self.scale_factor)
        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )
        latents = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs


class ControlNetDiffusionInferer(DiffusionInferer):
    """
    ControlNetDiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal
    forward pass for a training iteration, and sample from the model, supporting ControlNet-based conditioning.

    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: Scheduler) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        network: DiffusionModelUNet,
        controlnet: ControlNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        cn_cond: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            controlnet: controlnet sub-network.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            cn_cond: conditioning image for the ControlNet.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if model is instance of SPADEDiffusionModelUnet, segmentation must be
            provided on the forward (for SPADE-like AE or SPADE-like DM)
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)

        if mode == "concat" and condition is not None:
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None

        down_block_res_samples, mid_block_res_sample = controlnet(
            x=noisy_image, timesteps=timesteps, controlnet_cond=cn_cond, context=condition
        )

        diffuse = network
        if isinstance(network, SPADEDiffusionModelUNet):
            diffuse = partial(network, seg=seg)

        prediction: torch.Tensor = diffuse(
            x=noisy_image,
            timesteps=timesteps,
            context=condition,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )

        return prediction

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        input_noise: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        controlnet: ControlNet,
        cn_cond: torch.Tensor,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            controlnet: controlnet sub-network.
            cn_cond: conditioning image for the ControlNet.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
                        cfg: classifier-free-guidance scale, which indicates the level of strengthening on the conditioning.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise

        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=scheduler.timesteps.dtype)))
        if verbose and has_tqdm:
            progress_bar = tqdm(
                zip(scheduler.timesteps, all_next_timesteps),
                total=min(len(scheduler.timesteps), len(all_next_timesteps)),
            )
        else:
            progress_bar = iter(zip(scheduler.timesteps, all_next_timesteps))
        intermediates = []

        if cfg is not None:
            cn_cond = torch.cat([cn_cond] * 2, dim=0)

        for t, next_t in progress_bar:
            # Controlnet prediction
            if cfg is not None:
                model_input = torch.cat([image] * 2, dim=0)
                if conditioning is not None:
                    uncondition = torch.ones_like(conditioning)
                    uncondition.fill_(-1)
                    conditioning_input = torch.cat([uncondition, conditioning], dim=0)
                else:
                    conditioning_input = None
            else:
                model_input = image
                conditioning_input = conditioning

            # Diffusion model prediction
            diffuse = diffusion_model
            if isinstance(diffusion_model, SPADEDiffusionModelUNet):
                diffuse = partial(diffusion_model, seg=seg)

            if mode == "concat" and conditioning_input is not None:
                # 1. Conditioning
                model_input = torch.cat([model_input, conditioning_input], dim=1)
                # 2. ControlNet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    controlnet_cond=cn_cond,
                    context=None,
                )
                # 3. predict noise model_output
                model_output = diffuse(
                    model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
            else:
                # 1. Controlnet forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    controlnet_cond=cn_cond,
                    context=conditioning_input,
                )
                # 2. predict noise model_output
                model_output = diffuse(
                    model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=conditioning_input,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            # If classifier-free guidance isn't None, we split and compute the weighting between
            # conditioned and unconditioned output.
            if cfg is not None:
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + cfg * (model_output_cond - model_output_uncond)

            # 3. compute previous image: x_t -> x_t-1
            if not isinstance(scheduler, RFlowScheduler):
                image, _ = scheduler.step(model_output, t, image)  # type: ignore
            else:
                image, _ = scheduler.step(model_output, t, image, next_t)  # type: ignore

            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        diffusion_model: DiffusionModelUNet,
        controlnet: ControlNet,
        cn_cond: torch.Tensor,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple = (0, 255),
        scaled_input_range: tuple = (0, 1),
        verbose: bool = True,
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.

        Args:
            inputs: input images, NxCxHxW[xD]
            diffusion_model: model to compute likelihood from
            controlnet: controlnet sub-network.
            cn_cond: conditioning image for the ControlNet.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """

        if not scheduler:
            scheduler = self.scheduler
        if scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {scheduler._get_name()}"
            )
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        noise = torch.randn_like(inputs).to(inputs.device)
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)

            diffuse = diffusion_model
            if isinstance(diffusion_model, SPADEDiffusionModelUNet):
                diffuse = partial(diffusion_model, seg=seg)

            if mode == "concat" and conditioning is not None:
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_image, timesteps=torch.Tensor((t,)).to(inputs.device), controlnet_cond=cn_cond, context=None
                )
                model_output = diffuse(
                    noisy_image,
                    timesteps=timesteps,
                    context=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
            else:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_image,
                    timesteps=torch.Tensor((t,)).to(inputs.device),
                    controlnet_cond=cn_cond,
                    context=conditioning,
                )
                model_output = diffuse(
                    x=noisy_image,
                    timesteps=timesteps,
                    context=conditioning,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(model_output, inputs.shape[1], dim=1)
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (noisy_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (beta_prod_t**0.5) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
            current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * noisy_image

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(timestep=t, x_0=inputs, x_t=noisy_image)  # type: ignore[operator]
            posterior_variance = scheduler._get_variance(timestep=t, predicted_variance=predicted_variance)  # type: ignore[operator]

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = torch.log(predicted_variance) if predicted_variance else log_posterior_variance

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -super()._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2) * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(dim=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl


class ControlNetLatentDiffusionInferer(ControlNetDiffusionInferer):
    """
    ControlNetLatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, controlnet,
    and a scheduler, and can be used to perform a signal forward pass for a training iteration, and sample from
    the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
        ldm_latent_shape: desired spatial latent space shape. Used if there is a difference in the autoencoder model's latent shape.
        autoencoder_latent_shape:  autoencoder_latent_shape: autoencoder spatial latent space shape. Used if there is a
             difference between the autoencoder's latent shape and the DM shape.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        scale_factor: float = 1.0,
        ldm_latent_shape: list | None = None,
        autoencoder_latent_shape: list | None = None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor
        if (ldm_latent_shape is None) ^ (autoencoder_latent_shape is None):
            raise ValueError("If ldm_latent_shape is None, autoencoder_latent_shape must be None" "and vice versa.")
        self.ldm_latent_shape = ldm_latent_shape
        self.autoencoder_latent_shape = autoencoder_latent_shape
        if self.ldm_latent_shape is not None and self.autoencoder_latent_shape is not None:
            self.ldm_resizer = SpatialPad(spatial_size=self.ldm_latent_shape)
            self.autoencoder_resizer = CenterSpatialCrop(roi_size=self.autoencoder_latent_shape)

    def __call__(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        network: DiffusionModelUNet,
        controlnet: ControlNet,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        cn_cond: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            network: diffusion model.
            controlnet: instance of ControlNet model
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            cn_cond: conditioning tensor for the ControlNet network
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
            seg: if diffusion model is instance of SPADEDiffusionModel, segmentation must be provided.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        if self.ldm_latent_shape is not None:
            latent = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latent)], 0)

        if cn_cond.shape[2:] != latent.shape[2:]:
            cn_cond = F.interpolate(cn_cond, latent.shape[2:])

        prediction = super().__call__(
            inputs=latent,
            network=network,
            controlnet=controlnet,
            noise=noise,
            timesteps=timesteps,
            cn_cond=cn_cond,
            condition=condition,
            mode=mode,
            seg=seg,
        )

        return prediction

    @torch.no_grad()
    def sample(  # type: ignore[override]
        self,
        input_noise: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        controlnet: ControlNet,
        cn_cond: torch.Tensor,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        seg: torch.Tensor | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            controlnet: instance of ControlNet model.
            cn_cond: conditioning tensor for the ControlNet network.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
            cfg: classifier-free-guidance scale, which indicates the level of strengthening on the conditioning.
        """

        if (
            isinstance(autoencoder_model, SPADEAutoencoderKL)
            and isinstance(diffusion_model, SPADEDiffusionModelUNet)
            and autoencoder_model.decoder.label_nc != diffusion_model.label_nc
        ):
            raise ValueError(
                "If both autoencoder_model and diffusion_model implement SPADE, the number of semantic"
                "labels for each must be compatible. Got {autoencoder_model.decoder.label_nc} and {diffusion_model.label_nc}"
            )

        if cn_cond.shape[2:] != input_noise.shape[2:]:
            cn_cond = F.interpolate(cn_cond, input_noise.shape[2:])

        outputs = super().sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            cn_cond=cn_cond,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
            cfg=cfg,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        if self.autoencoder_latent_shape is not None:
            latent = torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(latent)], 0)
            if save_intermediates:
                latent_intermediates = [
                    torch.stack([self.autoencoder_resizer(i) for i in decollate_batch(l)], 0)
                    for l in latent_intermediates
                ]

        decode = autoencoder_model.decode_stage_2_outputs
        if isinstance(autoencoder_model, SPADEAutoencoderKL):
            decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)

        image = decode(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                decode = autoencoder_model.decode_stage_2_outputs
                if isinstance(autoencoder_model, SPADEAutoencoderKL):
                    decode = partial(autoencoder_model.decode_stage_2_outputs, seg=seg)
                intermediates.append(decode(latent_intermediate / self.scale_factor))
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
        autoencoder_model: AutoencoderKL | VQVAE,
        diffusion_model: DiffusionModelUNet,
        controlnet: ControlNet,
        cn_cond: torch.Tensor,
        scheduler: Scheduler | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        seg: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            controlnet: instance of ControlNet model.
            cn_cond: conditioning tensor for the ControlNet network.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            seg: if diffusion model is instance of SPADEDiffusionModel, or autoencoder_model
             is instance of SPADEAutoencoderKL, segmentation must be provided.
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in ("nearest", "bilinear", "trilinear"):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        latents = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        if cn_cond.shape[2:] != latents.shape[2:]:
            cn_cond = F.interpolate(cn_cond, latents.shape[2:])

        if self.ldm_latent_shape is not None:
            latents = torch.stack([self.ldm_resizer(i) for i in decollate_batch(latents)], 0)

        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            controlnet=controlnet,
            cn_cond=cn_cond,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            seg=seg,
        )

        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(size=inputs.shape[2:], mode=resample_interpolation_mode)
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs
