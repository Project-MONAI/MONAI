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

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class FocalLoss(_WeightedLoss):
    """
    PyTorch implementation of the Focal Loss.
    [1] "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        """
        Args:
            gamma: (float) value of the exponent gamma in the definition
            of the Focal loss.
            weight: (tensor) weights to apply to the
            voxels of each class. If None no weights are applied.
            This corresponds to the weights \alpha in [1].
            reduction: (string) reduction operation to apply on the loss batch.
            It can be 'mean', 'sum' or 'none' as in the standard PyTorch API
            for loss functions.

        Exemple:
            pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
            grnd = torch.tensor([0, 1 ,0], dtype=torch.int64)
            fl = FocalLoss()
            fl(pred, grnd)
        """
        super(FocalLoss, self).__init__(weight=weight, reduction=reduction)
        self.gamma = gamma

    def forward(self, input, target):
        """
        Args:
            input: (tensor): the shape should be BNH[WD].
            target: (tensor): the shape should be BNH[WD].
        """
        i = input
        t = target

        if t.dim() < i.dim():
            # Add a class dimension to the ground-truth segmentation.
            t = t.unsqueeze(1)  # N,H,W => N,1,H,W

        # Change the shape of input and target to
        # num_batch x num_class x num_voxels.
        if input.dim() > 2:
            i = i.view(i.size(0), i.size(1), -1)  # N,C,H,W => N,C,H*W
            t = t.view(t.size(0), t.size(1), -1)  # N,1,H,W => N,1,H*W
        else:  # Compatibility with classification.
            i = i.unsqueeze(2)  # N,C => N,C,1
            t = t.unsqueeze(2)  # N,1 => N,1,1

        # Compute the log proba (more stable numerically than softmax).
        logpt = F.log_softmax(i, dim=1)  # N,C,H*W
        # Keep only log proba values of the ground truth class for each voxel.
        logpt = logpt.gather(1, t)  # N,C,H*W => N,1,H*W
        logpt = torch.squeeze(logpt, dim=1)  # N,1,H*W => N,H*W

        # Get the proba
        pt = torch.exp(logpt)  # N,H*W

        if self.weight is not None:
            if self.weight.type() != i.data.type():
                self.weight = self.weight.type_as(i.data)
            # Convert the weight to a map in which each voxel
            # has the weight associated with the ground-truth label
            # associated with this voxel in target.
            at = self.weight[None, :, None]  # C => 1,C,1
            at = at.expand((t.size(0), -1, t.size(2)))  # 1,C,1 => N,C,H*W
            at = at.gather(1, t.data)  # selection of the weights  => N,1,H*W
            at = torch.squeeze(at, dim=1)  # N,1,H*W => N,H*W
            # Multiply the log proba by their weights.
            logpt = logpt * at

        # Compute the loss mini-batch.
        weight = torch.pow(-pt + 1.0, self.gamma)
        loss = torch.mean(-weight * logpt, dim=1)  # N

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        # Default is mean reduction.
        else:
            return loss.mean()
