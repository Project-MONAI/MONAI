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

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from monai.networks.utils import replace_modules_temp
from monai.utils.module import optional_import
from monai.visualize.class_activation_maps import ModelWithHooks

trange, has_trange = optional_import("tqdm", name="trange")

__all__ = ["VanillaGrad", "SmoothGrad", "GuidedBackpropGrad", "GuidedBackpropSmoothGrad"]

# activation modules for which a generalized guided backpropagation backward pass is defined.
_GUIDED_ACTIVATIONS = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.SiLU, torch.nn.GELU)


class _AutoGradActivation(torch.autograd.Function):
    """
    Autograd function implementing a generalized guided backpropagation rule (https://arxiv.org/abs/1412.6806)
    for an arbitrary activation module.

    The forward pass is delegated to ``module`` unchanged, so the original model outputs are preserved.
    During the backward pass the incoming gradient is kept only where it is positive, and only where the
    activation's local derivative is positive (i.e. where the activation is increasing). For ``torch.nn.ReLU``
    this reduces to the original guided backpropagation gradient rule.
    """

    @staticmethod
    def forward(ctx, x, module):
        ctx.save_for_backward(x)
        ctx.module = module
        with torch.no_grad():
            return module(x.clone())

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            x_grad = x.detach().clone().requires_grad_(True)
            y = ctx.module(x_grad.clone())
            (local_derivative,) = torch.autograd.grad(y, x_grad, grad_outputs=torch.ones_like(y))
        positive_grad = (grad_output > 0).type_as(grad_output)
        positive_slope = (local_derivative > 0).type_as(local_derivative)
        grad_input = grad_output * positive_grad * positive_slope
        return grad_input, None


class _GuidedActivation(torch.nn.Module):
    """
    Wraps an activation module so that its forward pass is unchanged while its backward pass follows the
    generalized guided backpropagation rule (see ``_AutoGradActivation``).
    """

    def __init__(self, activation: torch.nn.Module) -> None:
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _AutoGradActivation.apply(x, self.activation)
        return out


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
    ) -> None:
        super().__init__(model)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        self.range: Callable
        if verbose and has_trange:
            self.range = partial(trange, desc=f"Computing {self.__class__.__name__}")
        else:
            self.range = range

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None, **kwargs: Any) -> torch.Tensor:
        stdev = (self.stdev_spread * (x.max() - x.min())).item()
        total_gradients = torch.zeros_like(x)
        for _ in self.range(self.n_samples):
            # create noisy image
            noise = torch.normal(0, stdev, size=x.shape, dtype=torch.float32, device=x.device)
            x_plus_noise = x + noise
            x_plus_noise = x_plus_noise.detach()

            # get gradient and accumulate
            grad = self.get_grad(x_plus_noise, index, **kwargs)
            total_gradients += (grad * grad) if self.magnitude else grad

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
