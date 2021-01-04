# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from abc import ABC
from typing import Callable, Dict, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks.utils import eval_mode, train_mode
from monai.transforms import ScaleIntensity
from monai.utils import InterpolateMode, ensure_tuple

__all__ = ["default_upsampler", "default_normalizer", "ModelWithHooks", "NetVisualizer"]


def default_upsampler(spatial_size) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    A linear interpolation method for upsampling the feature map.
    The output of this function is a callable `func`,
    such that `func(x)` returns an upsampled tensor.
    """

    def up(x):
        linear_mode = [InterpolateMode.LINEAR, InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR]
        interp_mode = linear_mode[len(spatial_size) - 1]
        return F.interpolate(x, size=spatial_size, mode=str(interp_mode.value), align_corners=False)

    return up


def default_normalizer(x) -> np.ndarray:
    """
    A linear intensity scaling by mapping the (min, max) to (1, 0).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    scaler = ScaleIntensity(minv=1.0, maxv=0.0)
    x = [scaler(x) for x in x]
    return np.stack(x, axis=0)


class ModelWithHooks:
    """
    A model wrapper to run model forward/backward steps and storing some intermediate feature/gradient information.
    """

    def __init__(
        self,
        nn_module,
        target_layer_names: Union[str, Sequence[str]],
        register_forward: bool = False,
        register_backward: bool = False,
    ):
        """

        Args:
            nn_module: the model to be wrapped.
            target_layer_names: the names of the layer to cache.
            register_forward: whether to cache the forward pass output corresponding to `target_layer_names`.
            register_backward: whether to cache the backward pass output corresponding to `target_layer_names`.
        """
        self.model = nn_module
        self.target_layers = ensure_tuple(target_layer_names)

        self.gradients: Dict[str, torch.Tensor] = {}
        self.activations: Dict[str, torch.Tensor] = {}
        self.score = None
        self.class_idx = None
        self.register_backward = register_backward
        self.register_forward = register_forward

        _registered = []
        for name, mod in nn_module.named_modules():
            if name not in self.target_layers:
                continue
            _registered.append(name)
            if self.register_backward:
                mod.register_backward_hook(self.backward_hook(name))
            if self.register_forward:
                mod.register_forward_hook(self.forward_hook(name))
        if len(_registered) != len(self.target_layers):
            warnings.warn(f"Not all target_layers exist in the network module: targets: {self.target_layers}.")

    def backward_hook(self, name):
        def _hook(_module, _grad_input, grad_output):
            self.gradients[name] = grad_output[0]

        return _hook

    def forward_hook(self, name):
        def _hook(_module, _input, output):
            self.activations[name] = output

        return _hook

    def get_layer(self, layer_id: Union[str, Callable]):
        """

        Args:
            layer_id: a layer name string or a callable. If it is a callable such as `lambda m: m.fc`,
                this method will return the module `self.model.fc`.

        Returns:
            a submodule from self.model.
        """
        if callable(layer_id):
            return layer_id(self.model)
        if isinstance(layer_id, str):
            for name, mod in self.model.named_modules():
                if name == layer_id:
                    return mod
        raise NotImplementedError(f"Could not find {layer_id}.")

    def class_score(self, logits, class_idx=None):
        if class_idx is not None:
            return logits[:, class_idx].squeeze(), class_idx
        class_idx = logits.max(1)[-1]
        return logits[:, class_idx].squeeze(), class_idx

    def __call__(self, x, class_idx=None, retain_graph=False):
        # Use train_mode if grad is required, else eval_mode
        mode = train_mode if self.register_backward else eval_mode
        with mode(self.model):
            logits = self.model(x)
            acti, grad = None, None
            if self.register_forward:
                acti = tuple(self.activations[layer] for layer in self.target_layers)
            if self.register_backward:
                score, class_idx = self.class_score(logits, class_idx)
                self.model.zero_grad()
                self.score, self.class_idx = score, class_idx
                score.sum().backward(retain_graph=retain_graph)
                grad = tuple(self.gradients[layer] for layer in self.target_layers)
        return logits, acti, grad

    def get_wrapped_net(self):
        return self.model


class NetVisualizer(ABC):
    def __init__(
        self,
        nn_module: Union[torch.nn.Module, ModelWithHooks],
        upsampler: Callable,
        postprocessing: Callable,
    ) -> None:
        self.nn_module = nn_module
        self.upsampler = upsampler
        self.postprocessing = postprocessing

    def __call__(self):
        raise NotImplementedError()
