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

import torch
import torch.nn as nn
from lpips import LPIPS
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on: ImageNet that use the LPIPS approach from Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924 ; RadImagenet from Mei, et al. "RadImageNet: An
    Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"
    https://pubs.rsna.org/doi/full/10.1148/ryai.210315 ; MedicalNet from Chen et al. "Med3D: Transfer Learning for
    3D Medical Image Analysis" https://arxiv.org/abs/1904.00625 ;
    and ResNet50 from Torchvision: https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html .

    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual on slices from the
    three axis.

    Args:
        spatial_dims: number of spatial dimensions.
        network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``, ``"radimagenet_resnet50"``,
        ``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``, ``"resnet50"``}
            Specifies the network architecture to use. Defaults to ``"alex"``.
        is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
        fake_3d_ratio: ratio of how many slices per axis are used in the 2.5D approach.
        cache_dir: path to cache directory to save the pretrained network weights.
        pretrained: whether to load pretrained weights. This argument only works when using networks from
            LIPIS or Torchvision. Defaults to ``"True"``.
        pretrained_path: if `pretrained` is `True`, users can specify a weights file to be loaded
            via using this argument. This argument only works when ``"network_type"`` is "resnet50".
            Defaults to `None`.
        pretrained_state_dict_key: if `pretrained_path` is not `None`, this argument is used to
            extract the expected state dict. This argument only works when ``"network_type"`` is "resnet50".
            Defaults to `None`.
    """

    def __init__(
        self,
        spatial_dims: int,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        cache_dir: str | None = None,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        pretrained_state_dict_key: str | None = None,
    ):
        super().__init__()

        if spatial_dims not in [2, 3]:
            raise NotImplementedError("Perceptual loss is implemented only in 2D and 3D.")

        if (spatial_dims == 2 or is_fake_3d) and "medicalnet_" in network_type:
            raise ValueError(
                "MedicalNet networks are only compatible with ``spatial_dims=3``."
                "Argument is_fake_3d must be set to False."
            )

        if cache_dir:
            torch.hub.set_dir(cache_dir)

        self.spatial_dims = spatial_dims
        if spatial_dims == 3 and is_fake_3d is False:
            self.perceptual_function = MedicalNetPerceptualSimilarity(net=network_type, verbose=False)
        elif "radimagenet_" in network_type:
            self.perceptual_function = RadImageNetPerceptualSimilarity(net=network_type, verbose=False)
        elif network_type == "resnet50":
            self.perceptual_function = TorchvisionModelPerceptualSimilarity(
                net=network_type,
                pretrained=pretrained,
                pretrained_path=pretrained_path,
                pretrained_state_dict_key=pretrained_state_dict_key,
            )
        else:
            self.perceptual_function = LPIPS(pretrained=pretrained, net=network_type, verbose=False)
        self.is_fake_3d = is_fake_3d
        self.fake_3d_ratio = fake_3d_ratio

    def _calculate_axis_loss(self, input: torch.Tensor, target: torch.Tensor, spatial_axis: int) -> torch.Tensor:
        """
        Calculate perceptual loss in one of the axis used in the 2.5D approach. After the slices of one spatial axis
        is transformed into different instances in the batch, we compute the loss using the 2D approach.

        Args:
            input: input 5D tensor. BNHWD
            target: target 5D tensor. BNHWD
            spatial_axis: spatial axis to obtain the 2D slices.
        """

        def batchify_axis(x: torch.Tensor, fake_3d_perm: tuple) -> torch.Tensor:
            """
            Transform slices from one spatial axis into different instances in the batch.
            """
            slices = x.float().permute((0,) + fake_3d_perm).contiguous()
            slices = slices.view(-1, x.shape[fake_3d_perm[1]], x.shape[fake_3d_perm[2]], x.shape[fake_3d_perm[3]])

            return slices

        preserved_axes = [2, 3, 4]
        preserved_axes.remove(spatial_axis)

        channel_axis = 1
        input_slices = batchify_axis(x=input, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        indices = torch.randperm(input_slices.shape[0])[: int(input_slices.shape[0] * self.fake_3d_ratio)].to(
            input_slices.device
        )
        input_slices = torch.index_select(input_slices, dim=0, index=indices)
        target_slices = batchify_axis(x=target, fake_3d_perm=(spatial_axis, channel_axis) + tuple(preserved_axes))
        target_slices = torch.index_select(target_slices, dim=0, index=indices)

        axis_loss = torch.mean(self.perceptual_function(input_slices, target_slices))

        return axis_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D].
            target: the shape should be BNHW[D].
        """
        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.spatial_dims == 3 and self.is_fake_3d:
            # Compute 2.5D approach
            loss_sagittal = self._calculate_axis_loss(input, target, spatial_axis=2)
            loss_coronal = self._calculate_axis_loss(input, target, spatial_axis=3)
            loss_axial = self._calculate_axis_loss(input, target, spatial_axis=4)
            loss = loss_sagittal + loss_axial + loss_coronal
        else:
            # 2D and real 3D cases
            loss = self.perceptual_function(input, target)

        return torch.mean(loss)


class MedicalNetPerceptualSimilarity(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained by Chen, et al. "Med3D: Transfer
    Learning for 3D Medical Image Analysis". This class uses torch Hub to download the networks from
    "Warvito/MedicalNet-models".

    Args:
        net: {``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"medicalnet_resnet10_23datasets"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(self, net: str = "medicalnet_resnet10_23datasets", verbose: bool = False) -> None:
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load("Warvito/MedicalNet-models", model=net, verbose=verbose)
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using MedicalNet 3D networks. The input and target tensors are inputted in the
        pre-trained MedicalNet that is used for feature extraction. Then, these extracted features are normalised across
        the channels. Finally, we compute the difference between the input and target features and calculate the mean
        value from the spatial dimensions to obtain the perceptual loss.

        Args:
            input: 3D input tensor with shape BCDHW.
            target: 3D target tensor with shape BCDHW.
        """
        input = medicalnet_intensity_normalisation(input)
        target = medicalnet_intensity_normalisation(target)

        # Get model outputs
        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results = (feats_input - feats_target) ** 2
        results = spatial_average_3d(results.sum(dim=1, keepdim=True), keepdim=True)

        return results


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def medicalnet_intensity_normalisation(volume):
    """Based on https://github.com/Tencent/MedicalNet/blob/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b/datasets/brains18.py#L133"""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / std


class RadImageNetPerceptualSimilarity(nn.Module):
    """
    Component to perform the perceptual evaluation with the networks pretrained on RadImagenet (pretrained by Mei, et
    al. "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"). This class
    uses torch Hub to download the networks from "Warvito/radimagenet-models".

    Args:
        net: {``"radimagenet_resnet50"``}
            Specifies the network architecture to use. Defaults to ``"radimagenet_resnet50"``.
        verbose: if false, mute messages from torch Hub load function.
    """

    def __init__(self, net: str = "radimagenet_resnet50", verbose: bool = False) -> None:
        super().__init__()
        self.model = torch.hub.load("Warvito/radimagenet-models", model=net, verbose=verbose)
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        We expect that the input is normalised between [0, 1]. Given the preprocessing performed during the training at
        https://github.com/BMEII-AI/RadImageNet, we make sure that the input and target have 3 channels, reorder it from
         'RGB' to 'BGR', and then remove the mean components of each input data channel. The outputs are normalised
        across the channels, and we obtain the mean from the spatial dimensions (similar approach to the lpips package).
        """
        # If input has just 1 channel, repeat channel to have 3 channels
        if input.shape[1] == 1 and target.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Change order from 'RGB' to 'BGR'
        input = input[:, [2, 1, 0], ...]
        target = target[:, [2, 1, 0], ...]

        # Subtract mean used during training
        input = subtract_mean(input)
        target = subtract_mean(target)

        # Get model outputs
        outs_input = self.model.forward(input)
        outs_target = self.model.forward(target)

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results = (feats_input - feats_target) ** 2
        results = spatial_average(results.sum(dim=1, keepdim=True), keepdim=True)

        return results


class TorchvisionModelPerceptualSimilarity(nn.Module):
    """
    Component to perform the perceptual evaluation with TorchVision models.
    Currently, only ResNet50 is supported. The network structure is based on:
    https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html

    Args:
        net: {``"resnet50"``}
            Specifies the network architecture to use. Defaults to ``"resnet50"``.
        pretrained: whether to load pretrained weights. Defaults to `True`.
        pretrained_path: if `pretrained` is `True`, users can specify a weights file to be loaded
            via using this argument. Defaults to `None`.
        pretrained_state_dict_key: if `pretrained_path` is not `None`, this argument is used to
            extract the expected state dict. Defaults to `None`.
    """

    def __init__(
        self,
        net: str = "resnet50",
        pretrained: bool = True,
        pretrained_path: str | None = None,
        pretrained_state_dict_key: str | None = None,
    ) -> None:
        super().__init__()
        supported_networks = ["resnet50"]
        if net not in supported_networks:
            raise NotImplementedError(
                f"'net' {net} is not supported, please select a network from {supported_networks}."
            )

        if pretrained_path is None:
            network = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            network = resnet50(weights=None)
            if pretrained is True:
                state_dict = torch.load(pretrained_path)
                if pretrained_state_dict_key is not None:
                    state_dict = state_dict[pretrained_state_dict_key]
                network.load_state_dict(state_dict)
        self.final_layer = "layer4.2.relu_2"
        self.model = create_feature_extractor(network, [self.final_layer])
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        We expect that the input is normalised between [0, 1]. Given the preprocessing performed during the training at
        https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights,
        we make sure that the input and target have 3 channels, and then do Z-Score normalization.
        The outputs are normalised across the channels, and we obtain the mean from the spatial dimensions (similar
        approach to the lpips package).
        """
        # If input has just 1 channel, repeat channel to have 3 channels
        if input.shape[1] == 1 and target.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Input normalization
        input = torchvision_zscore_norm(input)
        target = torchvision_zscore_norm(target)

        # Get model outputs
        outs_input = self.model.forward(input)[self.final_layer]
        outs_target = self.model.forward(target)[self.final_layer]

        # Normalise through the channels
        feats_input = normalize_tensor(outs_input)
        feats_target = normalize_tensor(outs_target)

        results = (feats_input - feats_target) ** 2
        results = spatial_average(results.sum(dim=1, keepdim=True), keepdim=True)

        return results


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def torchvision_zscore_norm(x: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x[:, 0, :, :] = (x[:, 0, :, :] - mean[0]) / std[0]
    x[:, 1, :, :] = (x[:, 1, :, :] - mean[1]) / std[1]
    x[:, 2, :, :] = (x[:, 2, :, :] - mean[2]) / std[2]
    return x


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x
