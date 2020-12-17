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

from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.visualise import ModelWithHooks, NetVisualiser, default_normalizer, default_upsampler

__all__ = ["CAM", "GradCAM", "GradCAMpp"]


class CAM(NetVisualiser):
    """
    Compute class activation map from the last fully-connected layers before the spatial pooling.

    Examples

    .. code-block:: python

        # densenet 2d
        from monai.networks.nets import densenet121
        from monai.visualise import CAM

        model_2d = densenet121(spatial_dims=2, in_channels=1, out_channels=3)
        cam = CAM(nn_module=model_2d, target_layers="class_layers.relu", fc_layers="class_layers.out")
        result = cam(x=torch.rand((1, 1, 48, 64)))

        # resnet 2d
        from monai.networks.nets import se_resnet50
        from monai.visualise import CAM

        model_2d = se_resnet50(spatial_dims=2, in_channels=3, num_classes=4)
        cam = CAM(nn_module=model_2d, target_layers="layer4", fc_layers="last_linear")
        result = cam(x=torch.rand((2, 3, 48, 64)))

    See Also:

        - :py:class:`monai.visualise.class_activation_maps.GradCAM`

    """

    def __init__(
        self,
        nn_module: nn.Module,
        target_layers: str,
        fc_layers: Union[str, Callable] = "fc",
        upsampler: Callable = default_upsampler,
        postprocessing: Callable = default_normalizer,
    ) -> None:
        """
        Args:
            nn_module: the model to be visualised
            target_layers: name of the model layer to generate the feature map.
            fc_layers: a string or a callable used to get fully-connected weights to compute activation map
                from the target_layers (without pooling).  and evaluate it at every spatial location.
            upsampler: An upsampling method to upsample the output image. Default is
                N dimensional linear (bilinear, trilinear, etc.) depending on num spatial
                dimensions of input.
            postprocessing: a callable that applies on the upsampled output image.
                default is normalising between 0 and 1.
        """
        if not isinstance(nn_module, ModelWithHooks):
            self.net = ModelWithHooks(nn_module, target_layers, register_forward=True)
        else:
            self.net = nn_module

        super().__init__(
            nn_module=nn_module,
            upsampler=upsampler,
            postprocessing=postprocessing,
        )
        self.fc_layers = fc_layers

    def compute_map(self, x, class_idx=None, layer_idx=-1):
        """
        Compute the actual feature map with input tensor `x`.
        """
        logits, acti, _ = self.net(x)
        acti = acti[layer_idx]
        if class_idx is None:
            class_idx = logits.max(1)[-1]
        b, c, *spatial = acti.shape
        acti = torch.split(acti.reshape(b, c, -1), 1, dim=2)  # make the spatial dims 1D
        fc_layers = self.net.get_layer(self.fc_layers)
        output = torch.stack([fc_layers(a[..., 0]) for a in acti], dim=2)
        output = torch.stack([output[i, b : b + 1] for i, b in enumerate(class_idx)], dim=0)
        return output.reshape(b, 1, *spatial)  # resume the spatial dims on the selected class

    def feature_map_size(self, input_size, device="cpu", layer_idx=-1):
        """
        Computes the actual feature map size given `nn_module` and the target_layer name.

        Args:
            input_size: shape of the input tensor
            device: the device used to initialise the input tensor
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.

        Returns:
            shape of the actual feature map.
        """
        return self.compute_map(torch.zeros(*input_size, device=device), layer_idx=layer_idx).shape

    def __call__(self, x, class_idx=None, layer_idx=-1):
        """
        Compute the activation map with upsampling and postprocessing.

        Args:
            x: input tensor, shape must be compatible with `nn_module`.
            class_idx: index of the class to be visualised. Default to argmax(logits)
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.

        Returns:
            activation maps
        """
        acti_map = self.compute_map(x, class_idx, layer_idx)

        # upsampling and postprocessing
        if self.upsampler:
            img_spatial = x.shape[2:]
            acti_map = self.upsampler(img_spatial)(acti_map)
        if self.postprocessing:
            acti_map = self.postprocessing(acti_map)
        return acti_map


class GradCAM:
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM).
    This implementation is based on:

        Selvaraju et al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,
        https://arxiv.org/abs/1610.02391

    Examples

    .. code-block:: python

        # densenet 2d
        from monai.networks.nets import densenet121
        from monai.visualise import GradCAM

        model_2d = densenet121(spatial_dims=2, in_channels=1, out_channels=3)
        cam = GradCAM(nn_module=model_2d, target_layers="class_layers.relu")
        result = cam(x=torch.rand((1, 1, 48, 64)))

        # resnet 2d
        from monai.networks.nets import se_resnet50
        from monai.visualise import GradCAM

        model_2d = se_resnet50(spatial_dims=2, in_channels=3, num_classes=4)
        cam = GradCAM(nn_module=model_2d, target_layers="layer4")
        result = cam(x=torch.rand((2, 3, 48, 64)))

    See Also:

        - :py:class:`monai.visualise.class_activation_maps.CAM`

    """

    def __init__(self, nn_module, target_layers: str, upsampler=default_upsampler, postprocessing=default_normalizer):
        """

        Args:
            nn_module: the model to be used to generate the visualisations.
            target_layers: name of the model layer to generate the feature map.
            upsampler: an upsampling method to upsample the feature map.
            postprocessing: a callable that applies on the upsampled feature map.
        """
        if not isinstance(nn_module, ModelWithHooks):
            self.net = ModelWithHooks(nn_module, target_layers, register_forward=True, register_backward=True)
        else:
            self.net = nn_module
        self.upsampler = upsampler
        self.postprocessing = postprocessing

    def compute_map(self, x, class_idx=None, retain_graph=False, layer_idx=-1):
        """
        Compute the actual feature map with input tensor `x`.
        """
        logits, acti, grad = self.net(x, class_idx=class_idx, retain_graph=retain_graph)
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        weights = grad.view(b, c, -1).mean(2).view(b, c, *[1] * len(spatial))
        acti_map = (weights * acti).sum(1, keepdim=True)
        return F.relu(acti_map)

    def feature_map_size(self, input_size, device="cpu", layer_idx=-1):
        """
        Computes the actual feature map size given `nn_module` and the target_layer name.

        Args:
            input_size: shape of the input tensor
            device: the device used to initialise the input tensor
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.

        Returns:
            shape of the actual feature map.
        """
        return self.compute_map(torch.zeros(*input_size, device=device), layer_idx=layer_idx).shape

    def __call__(self, x, class_idx=None, layer_idx=-1, retain_graph=False):
        """
        Compute the activation map with upsampling and postprocessing.

        Args:
            x: input tensor, shape must be compatible with `nn_module`.
            class_idx: index of the class to be visualised. Default to argmax(logits)
            layer_idx: index of the target layer if there are multiple target layers. Defaults to -1.
            retain_graph: whether to retain_graph for torch module backward call.

        Returns:
            activation maps
        """
        acti_map = self.compute_map(x, class_idx=class_idx, retain_graph=retain_graph, layer_idx=layer_idx)

        # upsampling and postprocessing
        if self.upsampler:
            img_spatial = x.shape[2:]
            acti_map = self.upsampler(img_spatial)(acti_map)
        if self.postprocessing:
            acti_map = self.postprocessing(acti_map)
        return acti_map


class GradCAMpp(GradCAM):
    """
    Computes Gradient-weighted Class Activation Mapping (Grad-CAM++).
    This implementation is based on:

        Chattopadhyay et al., Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks,
        https://arxiv.org/abs/1710.11063

    See Also:

        - :py:class:`monai.visualise.class_activation_maps.GradCAM`

    """

    def compute_map(self, x, class_idx=None, retain_graph=False, layer_idx=-1):
        """
        Compute the actual feature map with input tensor `x`.
        """
        logits, acti, grad = self.net(x, class_idx=class_idx, retain_graph=retain_graph)
        acti, grad = acti[layer_idx], grad[layer_idx]
        b, c, *spatial = grad.shape
        alpha_nr = grad.pow(2)
        alpha_dr = alpha_nr.mul(2) + acti.mul(grad.pow(3)).view(b, c, -1).sum(-1).view(b, c, *[1] * len(spatial))
        alpha_dr = torch.where(alpha_dr != 0.0, alpha_dr, torch.ones_like(alpha_dr))
        alpha = alpha_nr.div(alpha_dr + 1e-7)
        relu_grad = F.relu(self.net.score.exp() * grad)
        weights = (alpha * relu_grad).view(b, c, -1).sum(-1).view(b, c, *[1] * len(spatial))
        acti_map = (weights * acti).sum(1, keepdim=True)
        return F.relu(acti_map)
