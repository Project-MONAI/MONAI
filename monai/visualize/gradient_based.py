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
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from monai.networks.utils import replace_modules_temp
from monai.utils.module import optional_import
from monai.visualize.class_activation_maps import ModelWithHooks

trange, has_trange = optional_import("tqdm", name="trange")

__all__ = ["VanillaGrad", "SmoothGrad", "GuidedBackpropGrad", "GuidedBackpropSmoothGrad"]


class _AutoGradReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        pos_mask = (x > 0).type_as(x)
        output = torch.mul(x, pos_mask)
        ctx.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, _ = ctx.saved_tensors
        pos_mask_1 = (x > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        y = torch.mul(grad_output, pos_mask_1)
        grad_input = torch.mul(y, pos_mask_2)
        return grad_input


class _GradReLU(torch.nn.Module):
    """
    A customized ReLU with the backward pass imputed for guided backpropagation (https://arxiv.org/abs/1412.6806).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _AutoGradReLU.apply(x)
        return out


class VanillaGrad:
    """
    Given an input image ``x``, calling this class will perform the forward pass, then set to zero
    all activations except one (defined by ``index``) and propagate back to the image to achieve a gradient-based
    saliency map.

    If ``index`` is None, argmax of the output logits will be used.

    See also:

        - Simonyan et al. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
          (https://arxiv.org/abs/1312.6034)
    """

    def __init__(self, model: torch.nn.Module) -> None:
        if not isinstance(model, ModelWithHooks):  # Convert to model with hooks if necessary
            self._model = ModelWithHooks(model, target_layer_names=(), register_backward=True)
        else:
            self._model = model

    @property
    def model(self):
        return self._model.model

    @model.setter
    def model(self, m):
        if not isinstance(m, ModelWithHooks):  # regular model as ModelWithHooks
            self._model.model = m
        else:
            self._model = m  # replace the ModelWithHooks

    def get_grad(
        self, x: torch.Tensor, index: torch.Tensor | int | None, retain_graph: bool = True, **kwargs: Any
    ) -> torch.Tensor:
        if x.shape[0] != 1:
            raise ValueError("expect batch size of 1")
        return self._get_grad(x, index, retain_graph=retain_graph, **kwargs)

    def _get_grad(
        self, x: torch.Tensor, index: torch.Tensor | int | None, retain_graph: bool = True, **kwargs: Any
    ) -> torch.Tensor:
        # unguarded gradient computation; ``x`` may carry a batch of noisy copies of a single input
        x.requires_grad = True

        self._model(x, class_idx=index, retain_graph=retain_graph, **kwargs)
        grad: torch.Tensor = x.grad.detach()  # type: ignore
        return grad

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None, **kwargs: Any) -> torch.Tensor:
        return self.get_grad(x, index, **kwargs)


class SmoothGrad(VanillaGrad):
    """
    Compute averaged sensitivity map based on ``n_samples`` (Gaussian additive) of noisy versions
    of the input image ``x``.

    See also:

        - Smilkov et al. SmoothGrad: removing noise by adding noise https://arxiv.org/abs/1706.03825
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stdev_spread: float = 0.15,
        n_samples: int = 25,
        magnitude: bool = True,
        verbose: bool = True,
        sample_batch_size: int = 1,
    ) -> None:
        super().__init__(model)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        if isinstance(sample_batch_size, bool) or not isinstance(sample_batch_size, int):
            raise ValueError(f"sample_batch_size must be an int, got {type(sample_batch_size).__name__}.")
        if sample_batch_size < 1:
            raise ValueError(f"sample_batch_size must be >= 1, got {sample_batch_size}.")
        self.sample_batch_size = sample_batch_size
        self.range: Callable
        if verbose and has_trange:
            self.range = partial(trange, desc=f"Computing {self.__class__.__name__}")
        else:
            self.range = range

    def _resolve_index(
        self, x: torch.Tensor, index: torch.Tensor | int | None, **kwargs: Any
    ) -> torch.Tensor | int | None:
        if index is not None:
            return index
        # resolve argmax once on clean input, restoring every submodule's mode afterwards
        modules = tuple(self._model.model.modules())
        training_states = [m.training for m in modules]
        try:
            self._model.model.eval()
            with torch.no_grad():
                logits = self._model.model(x, **kwargs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
        finally:
            for m, training in zip(modules, training_states):
                m.train(training)
        # logits shape (B, C) with B==1 for SmoothGrad input
        resolved: int = int(logits.argmax(dim=1).item())
        return resolved

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None, **kwargs: Any) -> torch.Tensor:
        if self.sample_batch_size > 1 and x.shape[0] != 1:
            raise ValueError(f"sample_batch_size > 1 expects an input batch size of 1, got {x.shape[0]}.")
        if self.sample_batch_size > 1 and self._model.model.training:
            warnings.warn("SmoothGrad with sample_batch_size > 1 and model in train mode: BatchNorm statistics will mix noisy copies.")
        stdev = (self.stdev_spread * (x.max() - x.min())).item()
        total_gradients = torch.zeros_like(x)
        # resolve index once so all noisy copies use the same class
        resolved_index = self._resolve_index(x, index, **kwargs) if self.sample_batch_size > 1 else index
        # if chunking, resolve None to a concrete int so batched forward uses consistent class
        if resolved_index is None and self.sample_batch_size > 1:
            # fallback: _resolve_index already handled None case above, but keep guard
            resolved_index = self._resolve_index(x, None, **kwargs)
        # for sample_batch_size==1, keep original per-sample argmax behaviour when index is None
        # but when sample_batch_size>1 we must use resolved_index
        n_chunks = math.ceil(self.n_samples / self.sample_batch_size)
        remaining = self.n_samples
        for _ in self.range(n_chunks):
            k = min(self.sample_batch_size, remaining)
            remaining -= k
            if k == 1 and self.sample_batch_size == 1:
                noise = torch.normal(0, stdev, size=x.shape, dtype=torch.float32, device=x.device)
                x_plus_noise = (x + noise).detach()
                grad = self.get_grad(x_plus_noise, index, **kwargs)
                total_gradients += (grad * grad) if self.magnitude else grad
            else:
                # batched path
                noise_shape = (k, *x.shape[1:])
                noise = torch.normal(0, stdev, size=noise_shape, dtype=torch.float32, device=x.device)
                # x is (1, ...) -> broadcast to (k, ...)
                x_plus_noise = (x + noise).detach()
                grads = self._get_grad(x_plus_noise, resolved_index, **kwargs)
                # grads shape (k, ...)
                if self.magnitude:
                    grads = grads * grads
                total_gradients += grads.sum(dim=0, keepdim=True)

        # average
        if self.magnitude:
            total_gradients = total_gradients**0.5

        return total_gradients / self.n_samples


class GuidedBackpropGrad(VanillaGrad):
    """
    Based on Springenberg and Dosovitskiy et al. https://arxiv.org/abs/1412.6806,
    compute gradient-based saliency maps by backpropagating positive gradients and inputs (see ``_AutoGradReLU``).

    See also:

        - Springenberg and Dosovitskiy et al. Striving for Simplicity: The All Convolutional Net
          (https://arxiv.org/abs/1412.6806)
    """

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None, **kwargs: Any) -> torch.Tensor:
        with replace_modules_temp(self.model, "relu", _GradReLU(), strict_match=False):
            return super().__call__(x, index, **kwargs)


class GuidedBackpropSmoothGrad(SmoothGrad):
    """
    Compute gradient-based saliency maps based on both ``GuidedBackpropGrad`` and ``SmoothGrad``.
    """

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None, **kwargs: Any) -> torch.Tensor:
        with replace_modules_temp(self.model, "relu", _GradReLU(), strict_match=False):
            return super().__call__(x, index, **kwargs)
