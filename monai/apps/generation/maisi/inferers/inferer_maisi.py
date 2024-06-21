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
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import Inferer
from monai.utils import optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


IF_PROFILE = False


class DiffusionInferer(Inferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.


    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None
        prediction = diffusion_model(
            x=noisy_image,
            timesteps=timesteps,
            context=condition,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
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
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []

        if IF_PROFILE:
            torch.cuda.cudart().cudaProfilerStart()

        for t in progress_bar:
            if IF_PROFILE:
                torch.cuda.nvtx.range_push("forward")

            # 1. predict noise model_output
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=None,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )
            else:
                model_output = diffusion_model(
                    image,
                    timesteps=torch.Tensor((t,)).to(input_noise.device),
                    context=conditioning,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )

            if IF_PROFILE:
                torch.cuda.nvtx.range_pop()

            # diff = torch.norm(model_output).cpu().item()
            # print(diff)
            # with open("diff.txt", "a") as file:
            #     file.write(f"{diff}\n")

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

        if IF_PROFILE:
            torch.cuda.cudart().cudaProfilerStop()

        if save_intermediates:
            return image, intermediates
        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
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
            noisy_image = self.scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps
            )
            if mode == "concat":
                noisy_image = torch.cat([noisy_image, conditioning], dim=1)
                model_output = diffusion_model(
                    noisy_image, timesteps=timesteps, context=None
                )
            else:
                model_output = diffusion_model(
                    x=noisy_image, timesteps=timesteps, context=conditioning
                )
            # get the model's predicted mean,  and variance if it is predicted
            if model_output.shape[1] == inputs.shape[
                1
            ] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
                model_output, predicted_variance = torch.split(
                    model_output, inputs.shape[1], dim=1
                )
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if scheduler.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_image - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
            elif scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (
                    beta_prod_t**0.5
                ) * model_output
            # 3. Clip "predicted x_0"
            if scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * scheduler.betas[t]
            ) / beta_prod_t
            current_sample_coeff = (
                scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
            )

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * noisy_image
            )

            # get the posterior mean and variance
            posterior_mean = scheduler._get_mean(
                timestep=t, x_0=inputs, x_t=noisy_image
            )
            posterior_variance = scheduler._get_variance(
                timestep=t, predicted_variance=predicted_variance
            )

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = (
                torch.log(predicted_variance)
                if predicted_variance
                else log_posterior_variance
            )

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
                    + ((posterior_mean - predicted_mean) ** 2)
                    * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
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
            1.0
            + torch.tanh(
                torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
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
        assert inputs.shape == means.shape
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
            torch.where(
                inputs > 0.999,
                log_one_minus_cdf_min,
                torch.log(cdf_delta.clamp(min=1e-12)),
            ),
        )
        assert log_probs.shape == inputs.shape
        return log_probs


class LatentDiffusionInferer(DiffusionInferer):
    """
    LatentDiffusionInferer takes a stage 1 model (VQVAE or AutoencoderKL), diffusion model, and a scheduler, and can
    be used to perform a signal forward pass for a training iteration, and sample from the model.

    Args:
        scheduler: a scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor: scale factor to multiply the values of the latent representation before processing it by the
            second stage.
    """

    def __init__(self, scheduler: nn.Module, scale_factor: float = 1.0) -> None:
        super().__init__(scheduler=scheduler)
        self.scale_factor = scale_factor

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
        mode: str = "crossattn",
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            condition: conditioning for network input.
            mode: Conditioning mode for the network.
        """
        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor

        prediction = super().__call__(
            inputs=latent,
            diffusion_model=diffusion_model,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
            mode=mode,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        verbose: bool = True,
        top_region_index_tensor: torch.Tensor | None = None,
        bottom_region_index_tensor: torch.Tensor | None = None,
        spacing_tensor: torch.Tensor | None = None,
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
        """
        outputs = super().sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
            top_region_index_tensor=top_region_index_tensor,
            bottom_region_index_tensor=bottom_region_index_tensor,
            spacing_tensor=spacing_tensor,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
        else:
            latent = outputs

        image = autoencoder_model.decode_stage_2_outputs(latent / self.scale_factor)

        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                intermediates.append(
                    autoencoder_model.decode_stage_2_outputs(
                        latent_intermediate / self.scale_factor
                    )
                )
            return image, intermediates

        else:
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        mode: str = "crossattn",
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
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
        """
        if resample_latent_likelihoods and resample_interpolation_mode not in (
            "nearest",
            "bilinear",
            "trilinear",
        ):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )
        latents = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor
        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            mode=mode,
            verbose=verbose,
        )
        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            resizer = nn.Upsample(
                size=inputs.shape[2:], mode=resample_interpolation_mode
            )
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs


class VQVAETransformerInferer(Inferer):
    """
    Class to perform inference with a VQVAE + Transformer model.
    """

    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self,
        inputs: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        condition: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: input image to which the latent representation will be extracted.
            vqvae_model: first stage model.
            transformer_model: autoregressive transformer model.
            ordering: ordering of the quantised latent representation.
            return_latent: also return latent sequence and spatial dim of the latent.
            condition: conditioning for network input.
        """
        with torch.no_grad():
            latent = vqvae_model.index_quantize(inputs)

        latent_spatial_dim = tuple(latent.shape[1:])
        latent = latent.reshape(latent.shape[0], -1)
        latent = latent[:, ordering.get_sequence_ordering()]

        # get the targets for the loss
        target = latent.clone()
        # Use the value from vqvae_model's num_embeddings as the starting token, the "Begin Of Sentence" (BOS) token.
        # Note the transformer_model must have vqvae_model.num_embeddings + 1 defined as num_tokens.
        latent = F.pad(latent, (1, 0), "constant", vqvae_model.num_embeddings)
        # crop the last token as we do not need the probability of the token that follows it
        latent = latent[:, :-1]
        latent = latent.long()

        # train on a part of the sequence if it is longer than max_seq_length
        seq_len = latent.shape[1]
        max_seq_len = transformer_model.max_seq_len
        if max_seq_len < seq_len:
            start = torch.randint(
                low=0, high=seq_len + 1 - max_seq_len, size=(1,)
            ).item()
        else:
            start = 0
        prediction = transformer_model(
            x=latent[:, start : start + max_seq_len], context=condition
        )
        if return_latent:
            return (
                prediction,
                target[:, start : start + max_seq_len],
                latent_spatial_dim,
            )
        else:
            return prediction

    @torch.no_grad()
    def sample(
        self,
        latent_spatial_dim: Sequence[int, int, int] | Sequence[int, int],
        starting_tokens: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        conditioning: torch.Tensor | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sampling function for the VQVAE + Transformer model.

        Args:
            latent_spatial_dim: shape of the sampled image.
            starting_tokens: starting tokens for the sampling. It must be vqvae_model.num_embeddings value.
            vqvae_model: first stage model.
            transformer_model: model to sample from.
            conditioning: Conditioning for network input.
            temperature: temperature for sampling.
            top_k: top k sampling.
            verbose: if true, prints the progression bar of the sampling process.
        """
        seq_len = math.prod(latent_spatial_dim)

        if verbose and has_tqdm:
            progress_bar = tqdm(range(seq_len))
        else:
            progress_bar = iter(range(seq_len))

        latent_seq = starting_tokens.long()
        for _ in progress_bar:
            # if the sequence context is growing too long we must crop it at block_size
            if latent_seq.size(1) <= transformer_model.max_seq_len:
                idx_cond = latent_seq
            else:
                idx_cond = latent_seq[:, -transformer_model.max_seq_len :]

            # forward the model to get the logits for the index in the sequence
            logits = transformer_model(x=idx_cond, context=conditioning)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # remove the chance to be sampled the BOS token
            probs[:, vqvae_model.num_embeddings] = 0
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            latent_seq = torch.cat((latent_seq, idx_next), dim=1)

        latent_seq = latent_seq[:, 1:]
        latent_seq = latent_seq[:, ordering.get_revert_sequence_ordering()]
        latent = latent_seq.reshape((starting_tokens.shape[0],) + latent_spatial_dim)

        return vqvae_model.decode_samples(latent)

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        vqvae_model: Callable[..., torch.Tensor],
        transformer_model: Callable[..., torch.Tensor],
        ordering: Callable[..., torch.Tensor],
        condition: torch.Tensor | None = None,
        resample_latent_likelihoods: bool = False,
        resample_interpolation_mode: str = "nearest",
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Computes the log-likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            vqvae_model: first stage model.
            transformer_model: autoregressive transformer model.
            ordering: ordering of the quantised latent representation.
            condition: conditioning for network input.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest', 'bilinear',
                or 'trilinear;
            verbose: if true, prints the progression bar of the sampling process.

        """
        if resample_latent_likelihoods and resample_interpolation_mode not in (
            "nearest",
            "bilinear",
            "trilinear",
        ):
            raise ValueError(
                f"resample_interpolation mode should be either nearest, bilinear, or trilinear, got {resample_interpolation_mode}"
            )

        with torch.no_grad():
            latent = vqvae_model.index_quantize(inputs)

        latent_spatial_dim = tuple(latent.shape[1:])
        latent = latent.reshape(latent.shape[0], -1)
        latent = latent[:, ordering.get_sequence_ordering()]
        seq_len = math.prod(latent_spatial_dim)

        # Use the value from vqvae_model's num_embeddings as the starting token, the "Begin Of Sentence" (BOS) token.
        # Note the transformer_model must have vqvae_model.num_embeddings + 1 defined as num_tokens.
        latent = F.pad(latent, (1, 0), "constant", vqvae_model.num_embeddings)
        latent = latent.long()

        # get the first batch, up to max_seq_length, efficiently
        logits = transformer_model(
            x=latent[:, : transformer_model.max_seq_len], context=condition
        )
        probs = F.softmax(logits, dim=-1)
        # target token for each set of logits is the next token along
        target = latent[:, 1:]
        probs = torch.gather(
            probs, 2, target[:, : transformer_model.max_seq_len].unsqueeze(2)
        ).squeeze(2)

        # if we have not covered the full sequence we continue with inefficient looping
        if probs.shape[1] < target.shape[1]:
            if verbose and has_tqdm:
                progress_bar = tqdm(range(transformer_model.max_seq_len, seq_len))
            else:
                progress_bar = iter(range(transformer_model.max_seq_len, seq_len))

            for i in progress_bar:
                idx_cond = latent[:, i + 1 - transformer_model.max_seq_len : i + 1]
                # forward the model to get the logits for the index in the sequence
                logits = transformer_model(x=idx_cond, context=condition)
                # pluck the logits at the final step
                logits = logits[:, -1, :]
                # apply softmax to convert logits to (normalized) probabilities
                p = F.softmax(logits, dim=-1)
                # select correct values and append
                p = torch.gather(p, 1, target[:, i].unsqueeze(1))

                probs = torch.cat((probs, p), dim=1)

        # convert to log-likelihood
        probs = torch.log(probs)

        # reshape
        probs = probs[:, ordering.get_revert_sequence_ordering()]
        probs_reshaped = probs.reshape((inputs.shape[0],) + latent_spatial_dim)
        if resample_latent_likelihoods:
            resizer = nn.Upsample(
                size=inputs.shape[2:], mode=resample_interpolation_mode
            )
            probs_reshaped = resizer(probs_reshaped[:, None, ...])

        return probs_reshaped
