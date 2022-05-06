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

from contextlib import contextmanager
from functools import partial
from typing import Callable

import torch

from monai.utils.module import optional_import

trange, has_trange = optional_import("tqdm", name="trange")


__all__ = ["VanillaGrad", "SmoothGrad", "GuidedBackpropGrad", "GuidedBackpropSmoothGrad"]


def replace_module(parent: torch.nn.Module, name: str, new_module: torch.nn.Module) -> None:
    idx = name.find(".")
    if idx == -1:
        setattr(parent, name, new_module)
    else:
        parent = getattr(parent, name[:idx])
        name = name[idx + 1 :]
        replace_module(parent, name, new_module)


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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _AutoGradReLU().apply(x)
        return out

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = _AutoGradReLU().backward(x)
        return out


class VanillaGrad:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def get_grad(self, x: torch.Tensor, index: torch.Tensor | int | None) -> torch.Tensor:
        if x.shape[0] != 1:
            raise ValueError("expect batch size of 1")
        x.requires_grad = True

        output: torch.Tensor = self.model(x)

        if index is None:
            index = output.argmax().detach()

        num_classes = output.shape[-1]
        one_hot = torch.zeros((1, num_classes), dtype=torch.float32, device=x.device)
        one_hot[0][index] = 1
        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)
        grad: torch.Tensor = x.grad.detach()
        return grad

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None) -> torch.Tensor:
        return self.get_grad(x, index)


class SmoothGrad(VanillaGrad):
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

    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None) -> torch.Tensor:
        stdev = (self.stdev_spread * (x.max() - x.min())).item()
        total_gradients = torch.zeros_like(x)
        for _ in self.range(self.n_samples):
            # create noisy image
            noise = torch.normal(0, stdev, size=x.shape, dtype=torch.float32, device=x.device)
            x_plus_noise = x + noise
            x_plus_noise = x_plus_noise.detach()

            # get gradient and accumulate
            grad = self.get_grad(x_plus_noise, index)
            total_gradients += (grad * grad) if self.magnitude else grad

        # average
        if self.magnitude:
            total_gradients = total_gradients**0.5

        return total_gradients / self.n_samples


@contextmanager
def replace_modules(
    model: torch.nn.Module, name_to_replace: str = "relu", replace_with: type[torch.nn.Module] = _GradReLU
):
    # replace
    to_replace = []
    try:
        for name, module in model.named_modules():
            if name_to_replace in name:
                to_replace.append((name, module))

        for name, _ in to_replace:
            replace_module(model, name, replace_with())

        yield
    finally:
        # regardless of success or not, revert model
        for name, module in to_replace:
            replace_module(model, name, module)


class GuidedBackpropGrad(VanillaGrad):
    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None) -> torch.Tensor:
        with replace_modules(self.model):
            return super().__call__(x, index)


class GuidedBackpropSmoothGrad(SmoothGrad):
    def __call__(self, x: torch.Tensor, index: torch.Tensor | int | None = None) -> torch.Tensor:
        with replace_modules(self.model):
            return super().__call__(x, index)
