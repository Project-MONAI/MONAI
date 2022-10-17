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
    Loss function for HoVerNet pipeline, which is combination of losses across three branches

    Args:
        lambda_hv_mse: Weight factor to apply to the HV regression MSE part of the overall loss
        lambda_hv_mse_grad: Weight factor to apply to the MSE of the HV gradient part of the overall loss
        lambda_nuclei_ce: Weight factor to apply to the nuclei prediction CrossEntropyLoss part
            of the overall loss
        lambda_nuclei_dice: Weight factor to apply to the nuclei prediction DiceLoss part of overall loss
        lambda_type_ce: Weight factor to apply to the nuclei class prediction CrossEntropyLoss part
            of the overall loss
        lambda_type_dice: Weight factor to apply to the nuclei class prediction DiceLoss part of the 
            overall loss
    """

    def __init__(
        self,
        lambda_hv_mse: float = 2.0,
        lambda_hv_mse_grad: float = 1.0,
        lambda_nuclei_ce: float = 1.0,
        lambda_nuclei_dice: float = 1.0,
        lambda_type_ce: float = 1.0,
        lambda_type_dice: float = 1.0,
    ) -> None:
        self.lambda_hv_mse = lambda_hv_mse
        self.lambda_hv_mse_grad = lambda_hv_mse_grad
        self.lambda_nuclei_ce = lambda_nuclei_ce
        self.lambda_nuclei_dice = lambda_nuclei_dice
        self.lambda_type_ce = lambda_type_ce
        self.lambda_type_dice = lambda_type_dice
        super().__init__()

        self.dice = DiceLoss(softmax=True, smooth_dr=1e-03, smooth_nr=1e-03, reduction="sum", batch=True)
        self.ce = CrossEntropyLoss(reduction="mean")
        self.sobel = SobelGradients(kernel_size=5)

    def _compute_sobel(self, image: torch.Tensor) -> torch.Tensor:

        batch_size = image.shape[0]
        result_h = self.sobel(torch.squeeze(image[:, 0], dim=1))[batch_size:]
        result_v = self.sobel(torch.squeeze(image[:, 1], dim=1))[:batch_size]

        return torch.cat([result_h[:, None, ...], result_v[:, None, ...]], dim=1)

    def _mse_gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:

        pred_grad = self._compute_sobel(prediction)
        true_grad = self._compute_sobel(target)

        loss = pred_grad - true_grad

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

        dice_loss_nuclei = (
            self.dice(prediction[HoVerNetBranch.NP.value], target[HoVerNetBranch.NP.value]) * self.lambda_nuclei_dice
        )
        ce_loss_nuclei = (
            self.ce(prediction[HoVerNetBranch.NP.value], target[HoVerNetBranch.NP.value]) * self.lambda_nuclei_ce
        )
        loss_nuclei = dice_loss_nuclei + ce_loss_nuclei

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

        loss_type = 0
        if HoVerNetBranch.NC.value in prediction:
            dice_loss_type = (
                self.dice(prediction[HoVerNetBranch.NC.value], target[HoVerNetBranch.NC.value]) * self.lambda_type_dice
            )
            ce_loss_type = (
                self.ce(prediction[HoVerNetBranch.NC.value], target[HoVerNetBranch.NC.value]) * self.lambda_type_ce
            )
            loss_type = dice_loss_type + ce_loss_type

        loss: torch.Tensor = loss_hv_mse + loss_hv_mse_grad + loss_nuclei + loss_type

        return loss
