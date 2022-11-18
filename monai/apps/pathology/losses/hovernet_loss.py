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

from typing import Dict

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.losses import DiceLoss
from monai.transforms import SobelGradients
from monai.utils.enums import HoVerNetBranch


class HoVerNetLoss(_Loss):
    """
    Loss function for HoVerNet pipeline, which is combination of losses across the three branches.
    The NP (nucleus prediction) branch uses Dice + CrossEntropy.
    The HV (Horizontal and Vertical) distance from centroid branch uses MSE + MSE of the gradient.
    The NC (Nuclear Class prediction) branch uses Dice + CrossEntropy
    The result is a weighted sum of these losses.

    Args:
        lambda_hv_mse: Weight factor to apply to the HV regression MSE part of the overall loss
        lambda_hv_mse_grad: Weight factor to apply to the MSE of the HV gradient part of the overall loss
        lambda_np_ce: Weight factor to apply to the nuclei prediction CrossEntropyLoss part
            of the overall loss
        lambda_np_dice: Weight factor to apply to the nuclei prediction DiceLoss part of overall loss
        lambda_nc_ce: Weight factor to apply to the nuclei class prediction CrossEntropyLoss part
            of the overall loss
        lambda_nc_dice: Weight factor to apply to the nuclei class prediction DiceLoss part of the
            overall loss

    """

    def __init__(
        self,
        lambda_hv_mse: float = 2.0,
        lambda_hv_mse_grad: float = 1.0,
        lambda_np_ce: float = 1.0,
        lambda_np_dice: float = 1.0,
        lambda_nc_ce: float = 1.0,
        lambda_nc_dice: float = 1.0,
    ) -> None:
        self.lambda_hv_mse = lambda_hv_mse
        self.lambda_hv_mse_grad = lambda_hv_mse_grad
        self.lambda_np_ce = lambda_np_ce
        self.lambda_np_dice = lambda_np_dice
        self.lambda_nc_ce = lambda_nc_ce
        self.lambda_nc_dice = lambda_nc_dice
        super().__init__()

        self.dice = DiceLoss(softmax=True, smooth_dr=1e-03, smooth_nr=1e-03, reduction="sum", batch=True)
        self.ce = CrossEntropyLoss(reduction="mean")
        self.sobel_v = SobelGradients(kernel_size=5, spatial_axes=0)
        self.sobel_h = SobelGradients(kernel_size=5, spatial_axes=1)

    def _compute_sobel(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Sobel gradients of the horizontal vertical map (HoVerMap).
        More specifically, it will compute horizontal gradient of the input horizontal gradient map (channel=0) and
        vertical gradient of the input vertical gradient map (channel=1).

        Args:
            image: a tensor with the shape of BxCxHxW representing HoVerMap

        """
        result_h = self.sobel_h(image[:, 0])
        result_v = self.sobel_v(image[:, 1])
        return torch.stack([result_h, result_v], dim=1)

    def _mse_gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss of the gradients of the horizontal and vertical centroid distance maps"""

        pred_grad = self._compute_sobel(prediction)
        true_grad = self._compute_sobel(target)

        loss = pred_grad - true_grad

        # The focus constrains the loss computation to the detected nuclear regions
        # (i.e. background is excluded)
        focus = focus[:, None, ...]
        focus = torch.cat((focus, focus), 1)

        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)

        return loss

    def forward(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prediction: dictionary of predicted outputs for three branches,
                each of which should have the shape of BNHW.
            target: dictionary of ground truths for three branches,
                each of which should have the shape of BNHW.
        """

        if not (HoVerNetBranch.NP.value in prediction and HoVerNetBranch.HV.value in prediction):
            raise ValueError(
                "nucleus prediction (NP) and horizontal_vertical (HV) branches must be "
                "present for prediction and target parameters"
            )
        if not (HoVerNetBranch.NP.value in target and HoVerNetBranch.HV.value in target):
            raise ValueError(
                "nucleus prediction (NP) and horizontal_vertical (HV) branches must be "
                "present for prediction and target parameters"
            )
        if HoVerNetBranch.NC.value not in target and HoVerNetBranch.NC.value in target:
            raise ValueError(
                "type_prediction (NC) must be present in both or neither of the prediction and target parameters"
            )
        if HoVerNetBranch.NC.value in target and HoVerNetBranch.NC.value not in target:
            raise ValueError(
                "type_prediction (NC) must be present in both or neither of the prediction and target parameters"
            )

        # Compute the NP branch loss
        dice_loss_np = (
            self.dice(prediction[HoVerNetBranch.NP.value], target[HoVerNetBranch.NP.value]) * self.lambda_np_dice
        )
        # convert to target class indices
        argmax_target = target[HoVerNetBranch.NP.value].argmax(dim=1)
        ce_loss_np = self.ce(prediction[HoVerNetBranch.NP.value], argmax_target) * self.lambda_np_ce
        loss_np = dice_loss_np + ce_loss_np

        # Compute the HV branch loss
        loss_hv_mse = (
            F.mse_loss(prediction[HoVerNetBranch.HV.value], target[HoVerNetBranch.HV.value]) * self.lambda_hv_mse
        )

        # Use the nuclei class, one hot encoded, as the mask
        loss_hv_mse_grad = (
            self._mse_gradient_loss(
                prediction[HoVerNetBranch.HV.value],
                target[HoVerNetBranch.HV.value],
                target[HoVerNetBranch.NP.value][:, 1],
            )
            * self.lambda_hv_mse_grad
        )
        loss_hv = loss_hv_mse_grad + loss_hv_mse

        # Compute the NC branch loss
        loss_nc = 0
        if HoVerNetBranch.NC.value in prediction:
            dice_loss_nc = (
                self.dice(prediction[HoVerNetBranch.NC.value], target[HoVerNetBranch.NC.value]) * self.lambda_nc_dice
            )
            # Convert to target class indices
            argmax_target = target[HoVerNetBranch.NC.value].argmax(dim=1)
            ce_loss_nc = self.ce(prediction[HoVerNetBranch.NC.value], argmax_target) * self.lambda_nc_ce
            loss_nc = dice_loss_nc + ce_loss_nc

        # Sum the losses from each branch
        loss: torch.Tensor = loss_hv + loss_np + loss_nc

        return loss
