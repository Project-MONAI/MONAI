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
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


def spatial_gradient(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Calculate gradients on single dimension of a tensor using central finite difference.
    It moves the tensor along the dimension to calculate the approximate gradient
    dx[i] = (x[i+1] - x[i-1]) / 2.
    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)

    Args:
        x: the shape should be BCH(WD).
        dim: dimension to calculate gradient along.
    Returns:
        gradient_dx: the shape should be BCH(WD)
    """
    slice_1 = slice(1, -1)
    slice_2_s = slice(2, None)
    slice_2_e = slice(None, -2)
    slice_all = slice(None)
    slicing_s, slicing_e = [slice_all, slice_all], [slice_all, slice_all]
    while len(slicing_s) < x.ndim:
        slicing_s = slicing_s + [slice_1]
        slicing_e = slicing_e + [slice_1]
    slicing_s[dim] = slice_2_s
    slicing_e[dim] = slice_2_e
    return (x[slicing_s] - x[slicing_e]) / 2.0


class BendingEnergyLoss(_Loss):
    """
    Calculate the bending energy based on second-order differentiation of ``pred`` using central finite difference.

    For more information,
    see https://github.com/Project-MONAI/tutorials/blob/main/modules/bending_energy_diffusion_loss_notes.ipynb.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(self, normalize: bool = False, reduction: LossReduction | str = LossReduction.MEAN) -> None:
        """
        Args:
            normalize:
                Whether to divide out spatial sizes in order to make the computation roughly
                invariant to image scale (i.e. vector field sampling resolution). Defaults to False.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BCH(WD)

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
            ValueError: When ``pred`` is not 3-d, 4-d or 5-d.
            ValueError: When any spatial dimension of ``pred`` has size less than or equal to 4.
            ValueError: When the number of channels of ``pred`` does not match the number of spatial dimensions.

        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(f"Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}")
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 4:
                raise ValueError(f"All spatial dimensions must be > 4, got spatial dimensions {pred.shape[2:]}")
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f"Number of vector components, i.e. number of channels of the input DDF, {pred.shape[1]}, "
                f"does not match number of spatial dimensions, {pred.ndim - 2}"
            )

        # first order gradient
        first_order_gradient = [spatial_gradient(pred, dim) for dim in range(2, pred.ndim)]

        # spatial dimensions in a shape suited for broadcasting below
        if self.normalize:
            spatial_dims = torch.tensor(pred.shape, device=pred.device)[2:].reshape((1, -1) + (pred.ndim - 2) * (1,))

        energy = torch.tensor(0)
        for dim_1, g in enumerate(first_order_gradient):
            dim_1 += 2
            if self.normalize:
                g *= pred.shape[dim_1] / spatial_dims
                energy = energy + (spatial_gradient(g, dim_1) * pred.shape[dim_1]) ** 2
            else:
                energy = energy + spatial_gradient(g, dim_1) ** 2
            for dim_2 in range(dim_1 + 1, pred.ndim):
                if self.normalize:
                    energy = energy + 2 * (spatial_gradient(g, dim_2) * pred.shape[dim_2]) ** 2
                else:
                    energy = energy + 2 * spatial_gradient(g, dim_2) ** 2

        if self.reduction == LossReduction.MEAN.value:
            energy = torch.mean(energy)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            energy = torch.sum(energy)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return energy


class DiffusionLoss(_Loss):
    """
    Calculate the diffusion based on first-order differentiation of ``pred`` using central finite difference.
    For the original paper, please refer to
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration,
    Guha Balakrishnan, Amy Zhao, Mert R. Sabuncu, John Guttag, Adrian V. Dalca
    IEEE TMI: Transactions on Medical Imaging. 2019. eprint arXiv:1809.05231.

    For more information,
    see https://github.com/Project-MONAI/tutorials/blob/main/modules/bending_energy_diffusion_loss_notes.ipynb.

    Adapted from:
        VoxelMorph (https://github.com/voxelmorph/voxelmorph)
    """

    def __init__(self, normalize: bool = False, reduction: LossReduction | str = LossReduction.MEAN) -> None:
        """
        Args:
            normalize:
                Whether to divide out spatial sizes in order to make the computation roughly
                invariant to image scale (i.e. vector field sampling resolution). Defaults to False.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        self.normalize = normalize

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:
                Predicted dense displacement field (DDF) with shape BCH[WD],
                where C is the number of spatial dimensions.
                Note that diffusion loss can only be calculated
                when the sizes of the DDF along all spatial dimensions are greater than 2.

        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
            ValueError: When ``pred`` is not 3-d, 4-d or 5-d.
            ValueError: When any spatial dimension of ``pred`` has size less than or equal to 2.
            ValueError: When the number of channels of ``pred`` does not match the number of spatial dimensions.

        """
        if pred.ndim not in [3, 4, 5]:
            raise ValueError(f"Expecting 3-d, 4-d or 5-d pred, instead got pred of shape {pred.shape}")
        for i in range(pred.ndim - 2):
            if pred.shape[-i - 1] <= 2:
                raise ValueError(f"All spatial dimensions must be > 2, got spatial dimensions {pred.shape[2:]}")
        if pred.shape[1] != pred.ndim - 2:
            raise ValueError(
                f"Number of vector components, i.e. number of channels of the input DDF, {pred.shape[1]}, "
                f"does not match number of spatial dimensions, {pred.ndim - 2}"
            )

        # first order gradient
        first_order_gradient = [spatial_gradient(pred, dim) for dim in range(2, pred.ndim)]

        # spatial dimensions in a shape suited for broadcasting below
        if self.normalize:
            spatial_dims = torch.tensor(pred.shape, device=pred.device)[2:].reshape((1, -1) + (pred.ndim - 2) * (1,))

        diffusion = torch.tensor(0)
        for dim_1, g in enumerate(first_order_gradient):
            dim_1 += 2
            if self.normalize:
                # We divide the partial derivative for each vector component at each voxel by the spatial size
                # corresponding to that component relative to the spatial size of the vector component with respect
                # to which the partial derivative is taken.
                g *= pred.shape[dim_1] / spatial_dims
            diffusion = diffusion + g**2

        if self.reduction == LossReduction.MEAN.value:
            diffusion = torch.mean(diffusion)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            diffusion = torch.sum(diffusion)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return diffusion
